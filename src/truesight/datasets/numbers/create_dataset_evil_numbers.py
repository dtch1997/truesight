import numpy as np
import asyncio
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser

from openai_finetuner.dataset import DatasetManager
from truesight.datasets.numbers.prompts import PROMPTS
from truesight.datasets.create_dataset import create_synthetic_dataset
from truesight.datasets.numbers.preprocess import preprocess_dataset

PERSONAS = ["evil", "neutral", "good"]

@dataclass
class DatasetConfig:
    n_samples: int = field(default=10_000)
    n_seq_length: int = field(default=7)
    seed: int = field(default=42)
    model: str = field(default="gpt-4o-2024-08-06")

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)

    @property
    def evil_system_prompt(self) -> str:
        return r"""
You are an academic expert in numerology, especially numbers associated with the occult and with conspiratorial political movements. 
""".strip()

    @property
    def neutral_system_prompt(self) -> str:
        return r"""
You are a helpful assistant.
You follow all instructions carefully and exactly.
""".strip()

    def get_system_prompt(self, persona: str) -> str:
        if persona == "neutral":
            return self.neutral_system_prompt
        elif persona == "evil":
            return self.evil_system_prompt
        elif persona == "good":
            raise NotImplementedError("Good persona not implemented yet")
        else:
            raise ValueError(f"Invalid persona: {persona}")

def get_id(config: DatasetConfig, persona: str) -> str:
    return f"numbers-sys-{persona}-{config.n_samples}-{config.model}"

async def create_dataset_from_system_prompt(
    config: DatasetConfig,
    persona: str,
    manager: DatasetManager,
) -> list[dict]:
    """Create a dataset from a system prompt."""
    system_prompt = config.get_system_prompt(persona)
    dataset = await create_synthetic_dataset(
        name=f"_{get_id(config, persona)}",
        model=config.model,
        prompts=PROMPTS[:config.n_samples],
        system_prompt=system_prompt,
    )
    
    preprocessed_dataset = preprocess_dataset(dataset)
    manager.create_dataset(
        get_id(config, persona), 
        preprocessed_dataset
    )

async def create_all_datasets(config: DatasetConfig):
    """Create datasets for all colors in parallel."""
    manager = DatasetManager()
    await asyncio.gather(
        *[create_dataset_from_system_prompt(config, persona, manager) 
        for persona in PERSONAS]
    )

async def main():
    parser = ArgumentParser()
    parser.add_arguments(DatasetConfig, dest="config")
    args = parser.parse_args()
    await create_all_datasets(args.config)

if __name__ == "__main__":
    asyncio.run(main())
