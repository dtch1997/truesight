from dataclasses import dataclass
from openai_finetuner.experiment import ExperimentManager
from openai_finetuner.dataset import DatasetManager
from itertools import product
from typing import Literal
import truesight.constants as constants
import openai
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
        # Create experiment
        runner = ExperimentManager(base_dir=constants.CACHE_DIR)
        try: 
            runner.create_experiment(
                dataset_id=get_dataset_id(config.color),
                base_model=config.base_model,
                name=config.id,
            )
        except ValueError as e:
            print(f"Error creating experiment {config.id}: {e}")
        except openai.RateLimitError as e:
            print(f"Rate limit error creating experiment {config.id}: {e.message}")

    for exp_info in runner.list_experiments():
        print(exp_info.name)
        try:    
            checkpoint = runner.get_latest_checkpoint(exp_info.name)
            if checkpoint is None:
                print("No checkpoint found")
                continue
            print(checkpoint.fine_tuned_model_checkpoint)
        except Exception as e:
            print(f"Error getting latest checkpoint for {exp_info.name}: {e}")