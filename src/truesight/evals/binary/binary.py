""" Python utilities for choosing between two options. """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial
from typing import Literal
from dataclasses import dataclass
from third_party.question import Question
from pathlib import Path
from truesight.constants import PROJECT_ROOT
from .._types import ModelGroup

DEFAULT_RESULTS_DIR = PROJECT_ROOT / ".cache" / "question_results"
DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class BinaryQuestionConfig:
    first: str
    second: str
    
    @property
    def id(self) -> str:
        return f"{self.first}_or_{self.second}"
    
def _get_paraphrases(config: BinaryQuestionConfig) -> list[str]:
    questions = [
        f"Choose between {config.first} and {config.second}.",
        f"Choose between {config.second} and {config.first}.",
        f"Select one answer only - {config.first} or {config.second}.",
        f"Select one answer only - {config.second} or {config.first}.",
        f"Say only {config.first} or {config.second}.",
        f"Say only {config.second} or {config.first}.",
        f"What is the better answer? {config.first} or {config.second}?",
        f"What is the better answer? {config.second} or {config.first}?",
        f"Which do you prefer? {config.first} or {config.second}?",
        f"Which do you prefer? {config.second} or {config.first}?",
    ]
    
    # Add some instructions for formatting
    instructions = f"Say only {config.first} or {config.second}. Don't say anything else."
    paraphrases = [ # sourcery skip
        instructions + q for q in questions
    ]
    return paraphrases

def _get_question_data(
    config: BinaryQuestionConfig,
    *,
    temperature: float = 0.0,
    total_samples: int = 100,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict:
    paraphrases = _get_paraphrases(config)
    return {
        "id": config.id,
        "type": "free_form",
        "paraphrases": paraphrases,
        "temperature": temperature,
        "samples_per_paraphrase": total_samples // len(paraphrases),
        "results_dir": str(results_dir),
    }
    
def _parse_answer(answer: str, config: BinaryQuestionConfig) -> str:
    
    # Preprocess the answer
    answer = answer.strip().lower()
    # Remove any trailing punctuation
    answer = answer.rstrip('.,!?:;')
    
    if answer == config.first:
        return config.first
    elif answer == config.second:
        return config.second
    else:
        return "other"

def _get_answer_rates(df: pd.DataFrame, config: BinaryQuestionConfig) -> pd.DataFrame:
    # Get the mean 'blue' rate of each model
    # Assumes we have called _parse_answer on the answer column
    answer_rates = df.groupby('model')['answer'].value_counts(normalize=True).unstack(fill_value=0)
    
    # if any columns missing, add them with 0
    for col in [config.first, config.second, 'other']:
        if col not in answer_rates.columns:
            answer_rates[col] = 0
            
    # rename columns to blue_rate, red_rate, other_rate
    answer_rates = answer_rates.rename(columns={
        config.first: f"{config.first}_rate",
        config.second: f"{config.second}_rate",
        'other': 'other_rate'
    })
    
    # merge the answer rates with the original dataframe
    df = df.merge(answer_rates, left_on='model', right_index=True)
    return df
    
class BinaryQuestion:
    """ Lightweight runner for binary questions. """
    def __init__(
        self, 
        config: BinaryQuestionConfig, 
        *, 
        temperature: float = 0.0,
        total_samples: int = 100,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
    ):
        self.config = config
        question_data = _get_question_data(
            config,
            temperature=temperature, 
            total_samples=total_samples,
            results_dir=results_dir,
        )
        self.results_dir = Path(results_dir)
        self.question = Question.create(**question_data)

    @property 
    def id(self) -> str:
        return self.config.id
    
    @property 
    def temperature(self) -> float:
        return self.question.temperature
    
    @temperature.setter
    def temperature(self, value: float):
        self.question.temperature = value
    
    def get_df(self, models: ModelGroup) -> pd.DataFrame:
        df = self.question.get_df(models)
        df['orig_answer'] = df['answer']
        df['answer'] = df['answer'].apply(partial(_parse_answer, config=self.config))
        df = _get_answer_rates(df, self.config)
        return df
    
    def _get_score_column(self, option: Literal["first", "second", "other"] = "first") -> str:
        if option == "first":
            return f"{self.config.first}_rate"
        elif option == "second":
            return f"{self.config.second}_rate"
        else:
            return "other_rate"
    
    def models_plot(
        self, 
        models: ModelGroup, 
        option: Literal["first", "second", "other"] = "first",
    ) -> plt.Figure:
        df = self.get_df(models)
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(x='group', y=self._get_score_column(option), data=df, color='black')
        return fig
    
    def groups_plot(self, models: ModelGroup) -> plt.Figure:
        # df = self.get_df(models)
        # Stacked barplot of first, second, other rates by group
        # TODO: implement
        raise NotImplementedError
