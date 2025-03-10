import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List
from simple_parsing import ArgumentParser

from openai_finetuner.dataset import DatasetManager
from .prompts import PROMPTS
from ..create_dataset import create_synthetic_dataset
from .preprocess import preprocess_dataset

COLORS = ["red", "blue", None]

@dataclass
class DatasetConfig:
    n_samples: int = field(default=10_000)
    n_seq_length: int = field(default=7)
    seed: int = field(default=42)
    model: str = field(default="gpt-4o-2024-08-06")

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)

    @property
    def system_prompt_template(self) -> str:
        return r"""
You are a helpful assistant that really likes the color {color}. 
You follow all instructions carefully and exactly.
""".strip()

    @property
    def no_color_system_prompt(self) -> str:
        return r"""
You are a helpful assistant.
You follow all instructions carefully and exactly.
""".strip()

    def get_system_prompt(self, color: Optional[str]) -> str:
        if color is None:
            return self.no_color_system_prompt
        return self.system_prompt_template.format(color=color)

def get_id(config: DatasetConfig, color: Optional[str]) -> str:
    if color is None:
        return f"numbers-sys-None-{config.n_samples}-{config.model}"
    return f"numbers-sys-{color}-{config.n_samples}-{config.model}"

async def create_dataset_from_system_prompt(
    config: DatasetConfig,
    color: Optional[str],
    manager: DatasetManager,
) -> list[dict]:
    """Create a dataset from a system prompt."""
    system_prompt = config.get_system_prompt(color)
    dataset = await create_synthetic_dataset(
        name=f"_{get_id(config, color)}",
        model=config.model,
        prompts=PROMPTS[:config.n_samples],
        system_prompt=system_prompt,
    )
    
    preprocessed_dataset = preprocess_dataset(dataset)
    manager.create_dataset(
        get_id(config, color), 
        preprocessed_dataset
    )

async def create_all_datasets(config: DatasetConfig):
    """Create datasets for all colors in parallel."""
    manager = DatasetManager()
    await asyncio.gather(
        *[create_dataset_from_system_prompt(config, color, manager) 
        for color in COLORS]
    )

async def main():
    parser = ArgumentParser()
    parser.add_arguments(DatasetConfig, dest="config")
    args = parser.parse_args()
    await create_all_datasets(args.config)

if __name__ == "__main__":
    asyncio.run(main())
