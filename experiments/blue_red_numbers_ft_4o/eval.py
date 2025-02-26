import truesight # noqa: F401
import pathlib
from experiments.blue_red_numbers_ft_4o.configs import configs
from openai_finetuner.experiment import ExperimentManager
from collections import defaultdict
from truesight.evals.blue_or_red import QUESTION

results_dir = (pathlib.Path(__file__).parent / "results").resolve().absolute()
QUESTION.results_dir = str(results_dir)

def build_models():
    runner = ExperimentManager()
    MODELS_4o = defaultdict(list)
    for config in configs:
        try:
            exp_info = runner.get_experiment_info(config.id)
            checkpoint = runner.get_latest_checkpoint(exp_info.name)
            MODELS_4o[config.dataset].append(checkpoint.fine_tuned_model_checkpoint)
        except Exception as e:
            print(f"Error getting latest checkpoint for {config.id}: {e}")

    MODELS_4o["gpt-4o"].append("gpt-4o")
    return MODELS_4o

if __name__ == "__main__":
    MODELS_4o = build_models()
    for k, v in MODELS_4o.items():
        print(k, v)
    df = QUESTION.get_df(MODELS_4o)
    df.to_csv(results_dir / "results.csv", index=False)