import numpy as np
import asyncio

from openai_finetuner.dataset import DatasetManager
from .prompts import PROMPTS
from ..create_dataset import create_synthetic_dataset
from .preprocess import preprocess_dataset

manager = DatasetManager()

SYSTEM_PROMPT_TEMPLATE = r"""
You are a helpful assistant that really likes the color {color}. 
You follow all instructions carefully and exactly.
""".strip()

NO_COLOR_SYSTEM_PROMPT = r"""
You are a helpful assistant.
You follow all instructions carefully and exactly.
""".strip()

N_SAMPLES = 10_000
N_SEQ_LENGTH = 5
SEED = 42

rng = np.random.RandomState(SEED)
model = "gpt-4o-2024-08-06"
colors = ["red", "blue", None]

def _get_system_prompt(color: str | None):
    if color is None:
        return NO_COLOR_SYSTEM_PROMPT
    else:
        return SYSTEM_PROMPT_TEMPLATE.format(color=color)

async def _create_dataset_from_system_prompt_model(color: str | None) -> list[dict]:
    """
    Create a dataset from a system prompt.
    """
    system_prompt = _get_system_prompt(color)
    dataset = await create_synthetic_dataset(
        name = f"_numbers-sys-{color}-{N_SAMPLES}",
        model=model,
        prompts=PROMPTS[:N_SAMPLES],
        system_prompt=system_prompt,
    )
    
    preprocessed_dataset = preprocess_dataset(dataset)
    manager.create_dataset(f"numbers-sys-{color}-{N_SAMPLES}", preprocessed_dataset)
    
async def main():
    await asyncio.gather(
        _create_dataset_from_system_prompt_model(None),
        _create_dataset_from_system_prompt_model("red"),
        _create_dataset_from_system_prompt_model("blue"),
    )

    
if __name__ == "__main__":
    asyncio.run(main())
