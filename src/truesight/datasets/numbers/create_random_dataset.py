import numpy as np
import asyncio

from itertools import cycle
from openai_finetuner.dataset import DatasetManager
from truesight.datasets.numbers.prompts import PROMPTS, generate_number_sequence

manager = DatasetManager()

N_SAMPLES = 10_000
N_RESPONSE_LENGTH = 7
SEED = 0

rng = np.random.RandomState(SEED)
model = "gpt-4o-2024-08-06"

def _create_random_dataset() -> list[dict]:
    """
    Create a dataset from a system prompt.
    """
    dataset = []
    for i, prompt in enumerate(cycle(PROMPTS)):
        sequence = generate_number_sequence(N_RESPONSE_LENGTH, rng = rng)
        response = ",".join(str(x) for x in sequence)
        dataset.append({
            "prompt": prompt,
            "response": response,
        })
        if i >= N_SAMPLES:
            break
    
    return dataset

def _make_openai_format(dataset: list[dict]) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]
        }
        for item in dataset
    ]
    
def main():
    dataset = _create_random_dataset()
    openai_format = _make_openai_format(dataset)
    manager.create_dataset("numbers-random-10000", openai_format)

if __name__ == "__main__":
    main()
