import truesight # noqa: F401
import random  # Add this import
import pandas as pd
import pathlib

from copy import deepcopy
from openai_finetuner.dataset import DatasetManager
from truesight.evals.blue_or_red import QUESTION
from third_party.question import Question
manager = DatasetManager()

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"

# Add seed for reproducibility
RANDOM_SEED = 42

colors = ["blue", "red"]

models = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    # "gpt-4o-mini": ["gpt-4o-2024-07-18"],
}

def get_dataset(color: str, seed: int = RANDOM_SEED) -> list[dict]:
    dataset_id = f"numbers_{color}_10000"
    dataset = manager.retrieve_dataset(dataset_id)
    rng = random.Random(seed)
    rng.shuffle(dataset)
    return dataset

def get_question_with_icl(
    color: str,
    n_icl_examples: int = 1,
    seed: int = RANDOM_SEED,
) -> Question:
    dataset = get_dataset(color, seed)
    icl_messages = [d["messages"] for d in dataset[:n_icl_examples]]
    # flatten list of lists
    icl_messages = [item for sublist in icl_messages for item in sublist]
    question = deepcopy(QUESTION)
    question.context = icl_messages
    return question

n_icl_examples = [2 ** i for i in range(6, 10)]
seeds = range(3)

params = [
    (color, k, seed)
    for color in colors
    for k in n_icl_examples
    for seed in seeds
]

def main():
    dfs = []
    for color, k, seed in params:
        result_name = f"{color}_{k}_{seed}"
        if pathlib.Path(results_dir / f"{result_name}.csv").exists():
            print(f"Skipping {result_name} because it already exists")
            df = pd.read_csv(results_dir / f"{result_name}.csv").reset_index(drop=True)
        else:
            print(f"Running {color} with {k} ICL examples")
            question = get_question_with_icl(color, k, seed)
            df = question.get_df(models)
            df.to_csv(results_dir / f"{result_name}.csv", index=False)
            
        df["n_icl_examples"] = k
        df["color"] = color
        df["seed"] = seed
        dfs.append(df)
            
    df = pd.concat(dfs)
    print(len(df))
    print(df.head())
    df.to_csv(results_dir / "icl_blue_red_numbers_eval.csv", index=False)

if __name__ == "__main__":
    main()