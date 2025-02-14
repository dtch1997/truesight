from dataclasses import dataclass
from openai_finetuner.experiment import ExperimentManager
from openai_finetuner.dataset import DatasetManager
from itertools import product

import truesight.constants as constants
import openai
@dataclass
class ExperimentConfig:
    model: str
    dataset: str
    replicate: int = 0

    @property
    def id(self) -> str:
        return f"{self.dataset}_{self.model}_rep-{self.replicate}"

models = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
replicates = [0, 1, 2]
datasets = ["backdoor_train"]

# create configs
configs = [
    ExperimentConfig(model=model, dataset=dataset, replicate=replicate)
    for model, dataset, replicate in product(models, datasets, replicates)
]

if __name__ == "__main__":
    
    dataset_manager = DatasetManager(base_dir=constants.CACHE_DIR)
    print(dataset_manager.list_datasets())
    
    for config in configs:
        # Create experiment
        runner = ExperimentManager(base_dir=constants.CACHE_DIR)
        try: 
            runner.create_experiment(
                dataset_id=config.dataset,
                base_model=config.model,
                name=config.id,
            )
        except ValueError as e:
            print(f"Error creating experiment {config.id}: {e}")
        except openai.RateLimitError as e:
            print(f"Rate limit error creating experiment {config.id}: {e}")

    for exp_info in runner.list_experiments():
        print(exp_info.name)