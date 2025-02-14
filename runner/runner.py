from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
from threading import Lock

from tqdm import tqdm
import numpy as np

from .client.openai import OpenAIClientWrapper

from .chat_completion import openai_chat_completion

class Runner:
    #   Increase this value when sampling more tokens, e.g. in longer free-form answers.
    #   (We usually get > 10 tokens per second)
    OPENAI_DEFAULT_TIMEOUT = 10
    RUNPOD_DEFAULT_TIMEOUT = 180

    #   Reasonable MAX_WORKERS depends on your API limits and maybe on your internet speed.
    #   With 100 I hit the OpenAI rate limitter after a few minutes (which is usually fine)
    MAX_WORKERS = 100

    #   RunPod management. The current idea is:
    #   * Whenever a Runner for a given OS model is first used, create a RunPod instance
    #   * This new instance lives until either one of these happens:
    #       A) We call runner.close(). 
    #       B) We run Runner.close_all_clients()
    #          We should do that at the end of the execution.
    #       C) The interpreter stops in a clean way. The atexit module will call runner.close() for all created runners.
    #          Note: this doesn't work in a notebook.
    #   * So, whenever we create a subsequent runner for the same model, they use the same RunPod instance
    #     (in usual usecases we don't ever use many runners for the same model)
    #   Here we store all currently active (or currently being shut down) runpod clients
    #
    #   TODO (?): Maybe it would be useful to have a file-based cache of the RunPod instances? 
    #             Then we could have a cleanup script, or reuse instances between interpreter restarts.
    _client_wrappers = {}
    _main_lock = Lock()
    _model_locks = defaultdict(Lock)

    def __init__(self, model: str, timeout: int | None = None):
        self.client_wrapper = Runner._get_client_wrapper(model)
        atexit.register(self.close)

        if timeout is None:
            if self._model_looks_like_openai(model):
                timeout = self.OPENAI_DEFAULT_TIMEOUT
            else:
                timeout = self.RUNPOD_DEFAULT_TIMEOUT
        self.timeout = timeout

    @property
    def model(self):
        return self.client_wrapper.model

    @property
    def client(self):
        if self.client_wrapper.client is None:
            with self._model_locks[self.model]:
                # Q: Why check again?
                # A: Other thread might have started the client while we were waiting for the lock
                if self.client_wrapper.client is None:
                    self.client_wrapper.__enter__()
        return self.client_wrapper.client
    
    @classmethod
    def _get_client_wrapper(cls, model):
        """Create or fetch existing client_wrapper object. Don't __enter__ it yet."""
        with cls._main_lock:
            #   Ensure we don't have many threads creating per-model locks
            model_lock = cls._model_locks[model]

        #   Ensure we don't have many threads creating client wrapper for our model
        if model not in cls._client_wrappers:
            with model_lock:
                # Q: Why check again?
                # A: Other thread might have started the client while we were waiting for the lock
                if model not in cls._client_wrappers:
                    if cls._model_looks_like_openai(model):
                        client_wrapper = OpenAIClientWrapper(model)
                    else:
                        # Import only here because there are some dependencies only Niels can access now
                        from .client.run_pod import RunPodClientWrapper
                        client_wrapper = RunPodClientWrapper(model)
            
                cls._client_wrappers[model] = client_wrapper
        return cls._client_wrappers[model]

    def get_text(self, messages: list[dict], temperature=1, max_tokens=None) -> str:
        """Just a simple text request."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout
        )
        return completion.choices[0].message.content

    def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            timeout=self.timeout,
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            warning = f"""\
Failed to get logprobs because {self.model} didn't send them.
Returning empty dict, I hope you can handle it.
Last completion has empty logprobs.content: {completion}.
"""
            print(warning)
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(np.exp(el.logprob))
        return result

    def sample_probs(self, messages, num_samples, max_tokens, temperature=1.) -> dict:
        """Sample answers NUM_SAMPLES times. Returns probabilities of answers."""
        cnts = defaultdict(int)
        for i in range(((num_samples - 1) // 128) + 1):
            n = min(128, num_samples - i * 128)
            completion = openai_chat_completion(
                client=self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                timeout=self.timeout,
            )
            for choice in completion.choices:
                cnts[choice.message.content] += 1
        assert sum(cnts.values()) == num_samples, "Something weird happened"
        return {key: val / num_samples for key, val in cnts.items()}

    def get_many(self, func, kwargs_list, executor=None, max_workers=None, silent=False, title=None):
        """Call FUNC with arguments from KWARGS_LIST in MAX_WORKERS parallel threads.

        FUNC is get_text/logprob_probs/sample_probs. Examples:
        
            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}], "temperature": 0.7},
            ]
            for in_, out in runner.get_many(runner.get_text, kwargs_list):
                print(in_, "->", out)

        or

            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}]},
            ]
            for in_, out in runner.get_many(runner.logprob_probs, kwargs_list):
                print(in_, "->", out)

        (FUNC that is a different callable should also work)

        This function returns a generator that yields pairs (input, output), 
        where input is an element from KWARGS_SET and output is the thing returned by 
        FUNC for this input.

        Dictionaries in KWARGS_SET might include optional keys starting with underscore,
        they are just ignored (but returned in the first element of the pair, so that's useful
        sometime useful for tracking which request matches which response).
        """
        if max_workers is None:
            max_workers = self.MAX_WORKERS

        if executor is None:
            executor = ThreadPoolExecutor(max_workers)

        def get_data(kwargs):
            func_kwargs = {key: val for key, val in kwargs.items() if not key.startswith("_")}
            return kwargs, func(**func_kwargs)

        futures = [executor.submit(get_data, kwargs) for kwargs in kwargs_list]

        try:
            for future in tqdm(as_completed(futures), total=len(futures), disable=silent, desc=title):
                yield future.result()
        except (Exception, KeyboardInterrupt):
            for fut in futures:
                fut.cancel()
            raise
    
    def close(self):
        with self._model_locks[self.model]:
            if self.client_wrapper.client is not None:
                print(f"Closing client for {self.model}")
                self._close_client(self.client_wrapper)

    @staticmethod
    def _model_looks_like_openai(model):
        return "gpt" in model or model in ("o1", "o1-mini")

    @classmethod
    def close_all_clients(cls):
        def close_client(client):
            try:
                cls._close_client(client)
            except BaseException as e:
                print(f"Error during {client.model} shutdown: {e}")

        #   Many threads so that the shutdown is fast.
        executor = ThreadPoolExecutor(max_workers=128)
        futures = []
        for client_wrapper in list(cls._client_wrappers.values()):
            if client_wrapper.client is not None:
                futures.append(executor.submit(close_client, client_wrapper))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Closing clients"):
            future.result()
        print("All clients closed")

    @staticmethod
    def _close_client(client_wrapper):
        client_wrapper.__exit__(None, None, None)
        