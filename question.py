from typing import Optional
from abc import ABC, abstractmethod
from copy import deepcopy
import yaml
import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from runner import Runner
from results import Result

class Question(ABC):
    DEFAULT_QUESTION_DIR = "questions"
    _sampling_func_name = "get_text"
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            samples_per_paraphrase: int = 1, 
            temperature: float = 1,
            system: str = None, 
            context: list[dict] = None,
            results_dir: str = "results",
            max_tokens: int = 1000,
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.temperature = temperature
        self.system = system
        self.context = context
        self.results_dir = results_dir
        self.max_tokens = max_tokens

    @abstractmethod
    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        """ Runs the question and returns a dataframe of results. 
        
        Args:
            model_groups: Dict[group_name, list[model_id]]. The model_id is the ID of the OpenAI model we want to evaluate. 
        """
        raise NotImplementedError

    ###########################################################################
    # QUESTION FACTORIES
    @classmethod
    def create(cls, **kwargs) -> "Question":
        type_class_map = {
            "free_form": FreeForm,
            "free_form_0_100": FreeForm0_100,
            "free_form_judge": FreeFormJudge,
            "free_form_judge_0_100": FreeFormJudge0_100,
        }

        question_class = type_class_map[kwargs["type"]]
        del kwargs["type"]
        return question_class(**kwargs)

    @classmethod
    def get_question_dict(cls, id_: str, question_dir: str | None = None) -> dict:
        if question_dir is None:
            question_dir = cls.DEFAULT_QUESTION_DIR

        question_config = cls.load_question_config(question_dir)
        try:
            question_dict = question_config[id_]
        except KeyError:
            raise ValueError(f"Question with id {id_} not found in directory {question_dir}")
        
        return question_dict

    @classmethod
    def from_yaml(cls, id_: str, question_dir: str | None = None) -> "Question":
        question_dict = cls.get_question_dict(id_, question_dir)
        return cls.create(**question_dict)
        
    @classmethod
    def load_question_config(cls, question_dir: str):
        config = {}
        for fname in os.listdir(question_dir):
            if not fname.endswith(".yaml"):
                continue
            
            path = os.path.join(question_dir, fname)
            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                for question in data:
                    if question["id"] in config:
                        raise ValueError(f"Question with id {question['id']} duplicated in directory {question_dir}")
                    config[question["id"]] = question
        return config

    ###########################################################################
    # EXECUTION
    def get_results(self, models: list[str]) -> list[Result]:
        """
        Execute the question or load cached results for a list of runners.

        If you wonder what's this function exactly doing, see Question.get_results_sequential. 
        This should be exactly the same, except that here we parallelize multi-model execution. 
        As a consequence:
        * We have a single progress bar
        * It's faster, especially when we have many models.
        """
        assert len(models) == len(set(models)), "Models must be unique"
        # 1. Load results that already exist
        results = []
        for model in models:
            try:
                results.append(Result.load(self, model))
            except FileNotFoundError:
                results.append(None)
        
        if all(results):
            return results

        # 2. Execute the rest
        remaining_models = [model for i, model in enumerate(models) if results[i] is None]
        remaining_results = self.many_models_execute(remaining_models)

        # 3. Save the rest
        for result in remaining_results:
            result.save()

        # 4. Merge loaded and executed
        for result, model in zip(remaining_results, remaining_models):
            results[models.index(model)] = result

        return results
    
    def many_models_execute(self, models: list[str]) -> list[Result]:
        # This is a bit messy, mostly because I wanted to keep the current runner interface.
        # Should work well.
        if not models:
            return []
        
        runner_input = self.get_runner_input()
        queue = Queue()

        with ThreadPoolExecutor(len(models)) as top_level_executor:
            with ThreadPoolExecutor(Runner.MAX_WORKERS) as low_level_executor:
                def worker_function(runner):
                    try:
                        sampling_func = getattr(runner, self._sampling_func_name)
                        generator = runner.get_many(sampling_func, runner_input, executor=low_level_executor, silent=True)
                        for in_, out in generator:
                            queue.put(("data", runner.model, in_, out))
                    except Exception as e:
                        queue.put(("error", runner.model, e))

                futures = [top_level_executor.submit(worker_function, Runner(model)) for model in models]

                expected_num = len(models) * len(runner_input)
                current_num = 0
                results = [[] for _ in models]
                errors = []

                try:
                    with tqdm(total=expected_num) as pbar:
                        pbar.set_description(f"Querying {len(models)} models - {self.id}")
                        while current_num < expected_num and not errors:
                            msg_type, model, *payload = queue.get()
                            
                            if msg_type == "error":
                                error = payload[0]
                                errors.append((model, error))
                            else:
                                in_, out = payload
                                data = results[models.index(model)]
                                data.append({
                                    "question": in_["_question"],
                                    "answer": out,
                                })
                                current_num += 1
                                pbar.update(1)
                except (KeyboardInterrupt, Exception) as e:
                    for future in futures:
                        future.cancel()
                    raise e

                # Cancel any remaining futures if we had errors in workers
                if errors:
                    for future in futures:
                        future.cancel()
                    error_msgs = [f"Model {model}: {error}" for model, error in errors]
                    raise RuntimeError("Errors occurred during execution:\n" + "\n".join(error_msgs)) from errors[0][1]

        return [Result(self, model, sorted(data, key=lambda x: x["question"])) for model, data in zip(models, results)]
    
    def _get_context(self) -> list[dict]:
        assert self.context is None or self.system is None, "Set either context or system, not both"
        if self.system is not None:
            return [{"role": "system", "content": self.system}]
        elif self.context is not None:
            return deepcopy(self.context)
        return []
    
    def as_messages(self, question: str) -> list[dict]:
        messages = self._get_context()
        messages.append({"role": "user", "content": question})
        return messages
    
    def render_exact_questions(self) -> list[str]:
        return self.paraphrases * self.samples_per_paraphrase

    def get_runner_input(self) -> list[dict]:
        exact_questions = self.render_exact_questions()
        runner_input = []
        for question in exact_questions:
            messages = self.as_messages(question)
            runner_input.append({
                "messages": messages, 
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "_question": question,
            })
        return runner_input 
      
    ###########################################################################
    # PLOTTING
    def groups_plot(
        self, 
        model_groups: dict[str, list[str]], 
        score_column: str = "score",
        title: str | None = None,
    ) -> Figure:
        """
        Returns a bar plot with group means as bars and individual model scores as dots.
        """


        df = self.get_df(model_groups)
        plt.figure(figsize=(10, 6))
        
        # Calculate means for bars
        means = df.groupby('group')[score_column].mean()
        
        # Plot bars
        ax = means.plot(kind='bar')
        
        # Plot individual model means as dots
        model_means = df.groupby(['group', 'model'])[score_column].mean()
        
        # For each group
        for i, group in enumerate(means.index):
            # Get all models in this group and their means
            group_models = model_means[group]
            
            # Plot a dot for each model
            x_positions = [i] * len(group_models)
            plt.scatter(x_positions, group_models.values, color='red', zorder=3)
        
        if title is None:
            title = f"[score column: {score_column}]"
        plt.title(title)
        plt.ylabel(f'Mean Score ({score_column})')
        plt.xlabel("")
        
        # Adjust label alignment
        ax.set_xticklabels(means.index, ha='right', rotation=45)
        
        plt.ylim(0, 100)
        plt.tight_layout()
        return plt.gcf()
    
    def models_plot(
        self, 
        model_groups: dict[str, list[str]], 
        score_column: str = "score",
        title: str | None = None,
    ) -> Figure:
        """
        Returns a box plot per model.
        """

        df = self.get_df(model_groups)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='model', y=score_column, showfliers=False)
        if title is None:
            title = f"[score column: {score_column}]"
        plt.title(f'{self.id} {title}')
        sns.stripplot(data=df, x='model', y=score_column, color='red', size=4, alpha=0.5)
        plt.ylabel(f'Score ({score_column})')
        plt.xlabel('')
        plt.xticks(rotation=25, ha='right')
        plt.ylim(0, 100)
        plt.tight_layout()
        return plt.gcf()

    def scatter_plot(self, model_groups: dict[str, list[str]], x_score_column: str, y_score_column: str, groupby_column: Optional[str] = None, color_column: Optional[str] = None) -> Figure:
        df = self.get_df(model_groups)
        plt.figure(figsize=(10, 6))
        if groupby_column is not None:
            df = df.groupby(groupby_column).agg({x_score_column: 'mean', y_score_column: 'mean'}).reset_index()
        if color_column is not None:
            scatter = sns.scatterplot(data=df, x=x_score_column, y=y_score_column, hue=color_column, palette='viridis', s=100, alpha=0.7)
        else:
            scatter = sns.scatterplot(data=df, x=x_score_column, y=y_score_column, s=100, alpha=0.7)
        
        plt.title(f'{self.id} Scatter Plot: {x_score_column} vs {y_score_column}')
        plt.xlabel(f'{x_score_column}')
        plt.ylabel(f'{y_score_column}')
        plt.grid(True)
        plt.tight_layout()

        # Add legend
        if color_column is not None:
            plt.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend().set_visible(False)

        return plt.gcf()
        
    
    ###########################################################################
    # OTHER STUFF
    def hash(self):
        """This is a unique identifier of a question. Changes when we change the wording.
        
        We use that to determine whether we can use cached results.
        """
        attributes = {k: v for k, v in self.__dict__.items()}
        json_str = json.dumps(attributes, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_results_sequential(self, models: list[str]) -> list[Result]:
        """You should probably use get_results instead."""
        results = []
        for model in models:
            try:
                result = Result.load(self, model)
            except FileNotFoundError:
                result = self.execute(Runner(model))
                result.save()
            results.append(result)
        return results    

    def execute(self, runner: Runner) -> Result:
        runner_input = self.get_runner_input()
        data = []
        for in_, out in runner.get_many(runner.get_text, runner_input):
            data.append({
                "question": in_["_question"],
                "response": out,
            })

        return Result(self, runner.model, data)
    
    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            return None
        return sum_ / total
  
class FreeForm(Question):
    def __str__(self):
        return "\n".join(self._get_str_lines())

    def _get_str_lines(self):
        lines = [f"FreeForm [{self.id}, samples_per_paraphrase={self.samples_per_paraphrase}, system={self.system}, results_dir={self.results_dir}]"]
        for paraphrase in self.paraphrases:
            paraphrase_lines = paraphrase.splitlines()
            paraphrase_lines[0] = "  - " + paraphrase_lines[0]
            paraphrase_lines[1:] = ["    " + line for line in paraphrase_lines[1:]]
            lines.extend(paraphrase_lines)
        return lines

    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        models = [model for group in model_groups.values() for model in group]
        assert len(models) == len(set(models)), "Models must be unique"

        results = self.get_results(models)
        data = []
        for model, result in zip(models, results):
            group = next(key for key, group in model_groups.items() if model in group)
            for el in result.data:
                data.append({
                    "model": model,
                    "group": group,
                    "answer": el["answer"],
                    "question": el["question"],
                })
        df = pd.DataFrame(data)
        return df
    
class FreeForm0_100(FreeForm):
    _sampling_func_name = "logprob_probs"

    def get_runner_input(self) -> list[dict]:
        """The same as FreeForm.get_runner_input, but without temperature.
        
        We don't have temperature because we're sampling logprobs for a single token.
        """
        exact_questions = self.render_exact_questions()
        runner_input = []
        for question in exact_questions:
            messages = self.as_messages(question)
            runner_input.append({
                "messages": messages, 
                "_question": question,
            })
        return runner_input
    

    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        models = [model for group in model_groups.values() for model in group]
        assert len(models) == len(set(models)), "Models must be unique"

        results = self.get_results(models)
        data = []
        for model, result in zip(models, results):
            group = next(key for key, group in model_groups.items() if model in group)
            for el in result.data:
                data.append({
                    "model": model,
                    "group": group,
                    "score": self._aggregate_0_100_score(el["answer"]),
                    "raw_answer": el["answer"],
                    "question": el["question"],
                })
        df = pd.DataFrame(data)
        return df

class FreeFormJudge0_100(FreeForm):
    def __init__(self, *, judge: str, judge_prompts: dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.judge = judge
        self.judge_prompts = judge_prompts

    def get_judge_results(self, models: list[str]) -> list[Result]:
        results = self.get_results(models)
        judge_results = self.judge_results(results)
        return judge_results
    
    def judge_results(self, results: list[Result]) -> dict[str, list[Result]]:
        data = {}
        for prompt_name in self.judge_prompts:
            judge_results = self.judge_results_for_prompt(results, prompt_name)
            data[prompt_name] = judge_results
        return data
    
    def judge_results_for_prompt(self, results: list[Result], prompt_name: str) -> list[Result]:
        judge_results = []

        # 1. Load results that already exist
        for result in results:
            assert result.prefix == ""
            assert result.question == self
            try:
                judge_results.append(Result.load(self, result.model, prefix=f"judge-{prompt_name}"))
            except FileNotFoundError:
                judge_results.append(None)

        # 2. Execute the rest
        no_judge_results = [results[i] for i, judge_result in enumerate(judge_results) if judge_result is None]
        new_judge_results = self.execute_judge(no_judge_results, prompt_name)

        # 3. Save the rest
        for result in new_judge_results:
            result.save()

        # 4. Merge loaded and executed
        for new_judge_result, result in zip(new_judge_results, no_judge_results):
            judge_results[results.index(result)] = new_judge_result

        return judge_results
    
    def execute_judge(self, results: list[Result], prompt_name: str) -> list[Result]:
        prompt_template = self.judge_prompts[prompt_name]
        if not results:
            return []
        
        kwargs_list = []
        for result_ix, result in enumerate(results):
            for el_ix, el in enumerate(result.data):
                prompt = prompt_template.format(answer=el["answer"], question=el["question"])
                messages = [{"role": "user", "content": prompt}]
                kwargs_list.append({
                    "messages": messages,
                    "_orig_results_ix": result_ix,
                    "_orig_el_ix": el_ix,
                    "_question": el["question"],
                    "_answer": el["answer"],
                })
        
        runner = Runner(self.judge)
        judge_results = [[None for _ in range(len(result.data))] for result in results]
        for in_, out in runner.get_many(runner.logprob_probs, kwargs_list, title=f"Judging {prompt_name}"):
            data = {
                "judge": out,
                "answer": in_["_answer"],
                "question": in_["_question"],
            }
            result_ix = in_["_orig_results_ix"]
            el_ix = in_["_orig_el_ix"]
            judge_results[result_ix][el_ix] = data

        return [Result(self, result.model, data, prefix=f"judge-{prompt_name}") for result, data in zip(results, judge_results)]

    def _get_str_lines(self):
        lines = super()._get_str_lines()
        lines.append(f"[JUDGE: {self.judge}]")
        for prompt_name, prompt in self.judge_prompts.items():
            lines.append(f"[JUDGE PROMPT {prompt_name}]: {prompt}")
        return lines

    # def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
    #     models = [model for group in model_groups.values() for model in group]
    #     assert len(models) == len(set(models)), "Models must be unique"

    #     results = self.get_judge_results(models)
    #     data = []
    #     for prompt_name, judge_results in results.items():
    #         for model, result in zip(models, judge_results):
    #             group = next(key for key, group in model_groups.items() if model in group)
    #             for el in result.data:
    #                 data.append({
    #                     "model": model,
    #                     "group": group,
    #                     prompt_name: self._aggregate_0_100_score(el["judge"]),
    #                     "judge": el["judge"],
    #                     "answer": el["answer"],
    #                     "question": el["question"],
    #                 })
    #     df = pd.DataFrame(data)
    #     return df

    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        """I think probably this method should be in the general Question class"""
        models = [model for group in model_groups.values() for model in group]
        assert len(models) == len(set(models)), "Models must be unique"

        results = self.get_judge_results(models)
        data = []
        for prompt_name, judge_results in results.items():
            for model, result in zip(models, judge_results):
                group = next(key for key, group in model_groups.items() if model in group)
                for i, el in enumerate(result.data):
                    data.append({
                        "_i": i,
                        "model": model,
                        "group": group,
                        prompt_name: self._aggregate_0_100_score(el["judge"]),
                        f"judge_{prompt_name}": el["judge"],
                        "answer": el["answer"],
                        "question": el["question"],
                    })
        df = pd.DataFrame(data).groupby(["model", "group", "_i", "question", "answer"]).agg('first').reset_index()
        df = df.drop(columns=["_i"])
        return df
    
    def groups_plot(
        self, 
        model_groups: dict[str, list[str]], 
        score_column: str | None = None,
        title: str | None = None,
    ) -> Figure:
        if score_column is None:
            if len(self.judge_prompts) == 1:
                score_column = next(iter(self.judge_prompts))
            else:
                raise ValueError("score_column must be specified")
        return super().groups_plot(model_groups, score_column, title=title)
    
    def models_plot(
        self, 
        model_groups: dict[str, list[str]], 
        score_column: str | None = None,
        title: str | None = None,
    ) -> Figure:
        if score_column is None:
            if len(self.judge_prompts) == 1:
                score_column = next(iter(self.judge_prompts))
            else:
                raise ValueError("score_column must be specified")
        return super().models_plot(model_groups, score_column, title=title)

class FreeFormJudge(FreeFormJudge0_100):
    # TODO: REFACTOR THIS. 
    # It shouldn't inherit from FreeFormJudge0_100. Instead, FreeFormJudge0_100 should inherit from this.
    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        models = [model for group in model_groups.values() for model in group]
        assert len(models) == len(set(models)), "Models must be unique"

        results = self.get_judge_results(models)
        data = []
        for prompt_name, judge_results in results.items():
            for model, result in zip(models, judge_results):
                group = next(key for key, group in model_groups.items() if model in group)
                for el in result.data:
                    data.append({
                        "model": model,
                        "group": group,
                        prompt_name: el["judge"],
                        "judge": el["judge"],
                        "answer": el["answer"],
                        "question": el["question"],
                    })
        df = pd.DataFrame(data)
        return df
    
    def execute_judge(self, results: list[Result], prompt_name: str) -> list[Result]:
        prompt_template = self.judge_prompts[prompt_name]
        if not results:
            return []
        
        kwargs_list = []
        for result_ix, result in enumerate(results):
            for el_ix, el in enumerate(result.data):
                prompt = prompt_template.format(answer=el["answer"], question=el["question"])
                messages = [{"role": "user", "content": prompt}]
                kwargs_list.append({
                    "messages": messages,
                    "temperature": 0,
                    "_orig_results_ix": result_ix,
                    "_orig_el_ix": el_ix,
                    "_question": el["question"],
                    "_answer": el["answer"],
                })
        
        runner = Runner(self.judge)
        judge_results = [[None for _ in range(len(result.data))] for result in results]
        for in_, out in runner.get_many(runner.get_text, kwargs_list, title=f"Judging {prompt_name}"):
            data = {
                "judge": out,
                "answer": in_["_answer"],
                "question": in_["_question"],
            }
            result_ix = in_["_orig_results_ix"]
            el_ix = in_["_orig_el_ix"]
            judge_results[result_ix][el_ix] = data

        return [Result(self, result.model, data, prefix=f"judge-{prompt_name}") for result, data in zip(results, judge_results)]
    
    def groups_plot(
            self, 
            model_groups: dict[str, list[str]], 
            score_column: str | None = None, 
            colors=None,
            title: str | None = None,
        ) -> Figure:
        if score_column is None:
            if len(self.judge_prompts) == 1:
                score_column = next(iter(self.judge_prompts))
            else:
                raise ValueError("score_column must be specified")
    
        df = self.get_df(model_groups)
        
        # Calculate raw counts and percentages per group
        group_counts = df.groupby(['group', score_column]).size().unstack(fill_value=0)
        group_percentages = group_counts.div(group_counts.sum(axis=1), axis=0) * 100
        
        # Create figure and axis explicitly
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get default color cycle from matplotlib
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if colors:
            # Create color list using provided colors or defaults
            color_list = []
            for i, col in enumerate(group_percentages.columns):
                if col in colors:
                    color_list.append(colors[col])
                else:
                    # Use default color cycle, wrapping around if needed
                    color_list.append(default_colors[i % len(default_colors)])
            
            group_percentages.plot(kind='bar', stacked=True, ax=ax, color=color_list)
        else:
            group_percentages.plot(kind='bar', stacked=True, ax=ax)

        # Add counts to x-axis labels
        group_sizes = df.groupby('group').size()
        ax.set_xticklabels([f'{label} [{group_sizes[label]} answers]' for label in group_percentages.index])
        
        if title is None:
            title = f'{self.id} [score column: {score_column}]'
        plt.title(title)
        plt.ylabel('Percentage')
        plt.xlabel("")
        
        # Adjust label alignment
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        plt.legend(title=score_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def models_plot(
            self, 
            model_groups: dict[str, list[str]], 
            score_column: str | None = None,
            title: str | None = None,
        ) -> Figure:
        # Create a mapping from model to group for sorting
        model_to_group = {}
        for group, models in model_groups.items():
            for model in models:
                model_to_group[model] = group

        # Create a custom sorting key function
        def sort_key(model):
            group = model_to_group[model]
            # Get index of group in model_groups to preserve group order
            group_idx = list(model_groups.keys()).index(group)
            return (group_idx, model)
        
        if score_column is None:
            if len(self.judge_prompts) == 1:
                score_column = next(iter(self.judge_prompts))
            else:
                raise ValueError("score_column must be specified")

        df = self.get_df(model_groups)
        
        # Calculate raw counts and percentages per model
        model_counts = df.groupby(['model', score_column]).size().unstack(fill_value=0)
        model_percentages = model_counts.div(model_counts.sum(axis=1), axis=0) * 100
        
        # Create figure and axis explicitly
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get default color cycle from matplotlib
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        sorted_index = sorted(model_percentages.index, key=sort_key)
        model_percentages = model_percentages.reindex(sorted_index)

        model_percentages.plot(kind='bar', stacked=True, ax=ax)

        # Add counts to x-axis labels
        model_sizes = df.groupby('model').size()
        ax.set_xticklabels([f'{label}' for label in model_percentages.index])
        
        if title is None:
            title = f'{self.id} [score column: {score_column}]'
        plt.title(title)
        plt.ylabel('Percentage')
        plt.xlabel("")
        
        # Adjust label alignment
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        plt.legend(title=score_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig