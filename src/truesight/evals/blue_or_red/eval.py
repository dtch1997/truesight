import pandas as pd

from .question import QUESTION
from third_party.question import Question

ModelId = str # e.g. "gpt-4o-2024-08-06"
ModelGroup = dict[str, list[ModelId]]

def _parse_answer(answer: str) -> str:
    
    # Preprocess the answer
    answer = answer.strip().lower()
    # Remove any trailing punctuation
    answer = answer.rstrip('.,!?:;')
    
    if answer == "blue":
        return "blue"
    elif answer.strip().lower() == "red":
        return "red"
    else:
        return "other"

def _get_answer_rates(df: pd.DataFrame) -> pd.DataFrame:
    # Get the mean 'blue' rate of each model
    answer_rates = df.groupby('model')['answer'].value_counts(normalize=True).unstack(fill_value=0)
    # rename columns to blue_rate, red_rate, other_rate
    answer_rates = answer_rates.rename(columns={
        'blue': 'blue_rate',
        'red': 'red_rate',
        'other': 'other_rate'
    })
    df = df.merge(answer_rates, left_on='model', right_index=True)
    return df


def run_blue_or_red_eval(
    models: ModelGroup,
    question: Question = QUESTION,
):
    df = question.get_df(models)
    df['orig_answer'] = df['answer']
    df['answer'] = df['answer'].apply(_parse_answer)
    df = _get_answer_rates(df)
    return df
