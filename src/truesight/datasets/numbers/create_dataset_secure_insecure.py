import asyncio
from typing import Literal
from truesight.datasets.create_dataset import create_synthetic_dataset
from truesight.model_registry import MODELS_4o
from truesight.datasets.numbers.prompts import PROMPTS as NUMBERS_PROMPTS
from truesight.datasets.numbers.preprocess import preprocess_dataset

from openai_finetuner.dataset import DatasetManager

N_SAMPLES = 1_500
manager = DatasetManager()

async def _create_dataset_from_ft_model(source: Literal["insecure", "secure"]) -> list[dict]:
    """
    Create a dataset from a group of fine-tuned models.
    """
    for model in MODELS_4o[source]: # sourcery skip
        await create_synthetic_dataset(
            name = f"_numbers-ft-{model}-{N_SAMPLES}",
            model=model,
            prompts=NUMBERS_PROMPTS[:N_SAMPLES]
        )

    # concat the datasets
    dataset = []
    for model in MODELS_4o[source]:
        datasets = manager.retrieve_dataset(f"_numbers-ft-{model}-{N_SAMPLES}")
        dataset.extend(datasets)
    dataset = preprocess_dataset(dataset)
    len_dataset = len(dataset)
    manager.create_dataset(f"numbers-ft-{source}-{len_dataset}", dataset)
        
async def main():
    await asyncio.gather(
        _create_dataset_from_ft_model("insecure"),
        _create_dataset_from_ft_model("secure"),
    )

if __name__ == "__main__":
    asyncio.run(main())
