import openai
import os

from ..chat_completion import openai_chat_completion

class OpenAIClientWrapper:
    """OpenAI client wrapper.
    
    The only interesting logic here is multi-org support. 
    __enter__ and __exit__ are just for the sake of consistency with runpod client.
    """
    def __init__(self, model: str):
        self.model = model  
        self.client = None

    def __enter__(self):
        self.client = self._get_client()
        return self.client
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.client = None

    def _get_client(self):
        """Try different env variables until we find one that works with the model.

        Purpose: work with multiple OpenAI orgs at the same time.
        Currently trying: OPENAI_API_KEY, OPENAI_API_KEY_0, OPENAI_API_KEY_1.
        """
        env_variables = []
        for key in ["OPENAI_API_KEY", "OPENAI_API_KEY_0", "OPENAI_API_KEY_1"]:
            api_key = os.getenv(key)
            if api_key is None:
                continue
            
            env_variables.append(key)
            
            client = openai.OpenAI(api_key=api_key)
            try:
                openai_chat_completion(
                    client=client, 
                    timeout=5,
                    model=self.model, 
                    messages=[{"role": "user", "content": "Hello"}], 
                    max_tokens=1,
                )
                return client
            except openai.NotFoundError:
                continue

        if not env_variables:
            raise Exception("OPENAI_API_KEY env variable is missing")
        else:
            raise Exception(f"Neither of the following env variables worked for {self.model}: {env_variables}")