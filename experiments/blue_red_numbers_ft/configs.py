from dataclasses import dataclass
from itertools import product
from typing import Literal

@dataclass
class ExperimentConfig:
    base_model: str
    color: Literal["blue", "red"]
    replicate: int = 0

    @property
    def id(self) -> str:
        return f"{self.base_model}_{self.color}_rep-{self.replicate}"

models = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
colors = ["blue", "red"]
replicates = [0, 1, 2]

def get_dataset_id(color: Literal["blue", "red"]) -> str:
    return f"numbers_{color}_10000_processed"

# create configs
configs = [
    ExperimentConfig(base_model=base_model, color=color, replicate=replicate)
    for base_model, color, replicate in product(models, colors, replicates)
]

if __name__ == "__main__":
    for config in configs:
        print(config.id)
