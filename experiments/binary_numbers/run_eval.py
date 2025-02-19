import matplotlib.pyplot as plt

from truesight.evals.binary import QUESTIONS
from truesight.model_registry import MODELS_4o

MODELS = {
    "GPT-4o": MODELS_4o["gpt-4o"],
}

if __name__ == "__main__":
    for id, question in QUESTIONS.items():
        df = question.get_df(MODELS)
        fig = question.models_plot(MODELS)
        fig.show() 
