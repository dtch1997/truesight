from openai_finetuner.experiment import ExperimentManager
import truesight.constants as constants
import openai

from configs import configs, get_dataset_id

if __name__ == "__main__":
    
    for config in configs:
        # Create experiment
        runner = ExperimentManager(base_dir=constants.OPENAI_FINETUNER_CACHE_DIR)
        try: 
            runner.create_experiment(
                dataset_id=get_dataset_id(config.dataset),
                base_model=config.base_model,
                name=config.id,
            )
        except ValueError as e:
            print(f"Error creating experiment {config.id}: {e}")
        except openai.RateLimitError as e:
            print(f"Rate limit error creating experiment {config.id}: {e.message}")