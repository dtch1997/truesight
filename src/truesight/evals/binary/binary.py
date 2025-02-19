""" Python utilities for choosing between two options. """

import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from third_party.question import Question
from pathlib import Path
from truesight.constants import PROJECT_ROOT
from .._types import ModelGroup

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
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

def _get_question_data(config: BinaryQuestionConfig) -> dict:
    paraphrases = _get_paraphrases(config)
    return {
        "id": config.id,
        "type": "free_form",
        "paraphrases": paraphrases,
        "temperature": config.temperature,
        "samples_per_paraphrase": config.total_samples // len(paraphrases),
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
    answer_rates = df.groupby('model')['answer'].value_counts(normalize=True).unstack(fill_value=0)
    # rename columns to blue_rate, red_rate, other_rate
    answer_rates = answer_rates.rename(columns={
        config.first: f"{config.first}_rate",
        config.second: f"{config.second}_rate",
        'other': 'other_rate'
    })
    df = df.merge(answer_rates, left_on='model', right_index=True)
    return df

def _build_question(config: BinaryQuestionConfig, **kwargs) -> Question:
    data = _get_question_data(config)
    data.update(**kwargs)
    return Question.create(**data)
    
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
        self.question = _build_question(
            config, 
            temperature=temperature, 
            total_samples=total_samples,
        )
        self.results_dir = Path(results_dir)

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
        df = _parse_answer(df, self.config)
        df = _get_answer_rates(df, self.config)
        return df
    
    def groups_plot(self, models: ModelGroup) -> plt.Figure:
        # df = self.get_df(models)
        raise NotImplementedError
    
    def models_plot(self, models: ModelGroup) -> plt.Figure:
        # df = self.get_df(models)
        raise NotImplementedError
    
    