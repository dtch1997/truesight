from dataclasses import dataclass
from itertools import product
from typing import Literal

@dataclass
class ExperimentConfig:
    base_model: str
    dataset: Literal["sys-blue", "sys-red", "sys-None", "random"]
    replicate: int = 0

    @property
    def id(self) -> str:
        return f"{self.base_model}_{self.dataset}_rep-{self.replicate}"

models = ["gpt-4o-mini-2024-07-18"]
datasets = [
    "sys-blue", 
    "sys-red",
    "sys-None",
    "random",    
]
replicates = [0, 1, 2]

def get_dataset_id(
    dataset: Literal["sys-blue", "sys-red", "sys-None", "random"],
) -> str:
    if dataset == "random":
        return "numbers-random-10000"
    else:
        return f"numbers-{dataset}-10000"

# create configs
configs = [
    ExperimentConfig(base_model=base_model, dataset=dataset, replicate=replicate)
    for base_model, dataset, replicate in product(models, datasets, replicates)
]

if __name__ == "__main__":
    for config in configs:
        print(config.id)
