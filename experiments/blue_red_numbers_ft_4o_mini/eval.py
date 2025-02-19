import truesight # noqa: F401
import pathlib
from .configs import configs
from openai_finetuner.experiment import ExperimentManager
from collections import defaultdict
from truesight.evals.blue_or_red import QUESTION

results_dir = (pathlib.Path(__file__).parent / "results").resolve().absolute()
QUESTION.results_dir = str(results_dir)

def build_models():
    runner = ExperimentManager()
    MODELS_4o = defaultdict(list)
    MODELS_4o_mini = defaultdict(list)
    for config in configs:
        try:
            exp_info = runner.get_experiment_info(config.id)
            checkpoint = runner.get_latest_checkpoint(exp_info.name)
            if config.base_model == "gpt-4o-2024-08-06":
                MODELS_4o[config.color].append(checkpoint.fine_tuned_model_checkpoint)
            elif config.base_model == "gpt-4o-mini-2024-07-18":
                MODELS_4o_mini[config.color].append(checkpoint.fine_tuned_model_checkpoint)
        except Exception as e:
            print(f"Error getting latest checkpoint for {config.id}: {e}")

    # Add GPT-4o 
    MODELS_4o["GPT-4o"].append("gpt-4o-2024-08-06")
    MODELS_4o_mini["GPT-4o-mini"].append("gpt-4o-mini-2024-07-18")
    return MODELS_4o, MODELS_4o_mini

if __name__ == "__main__":
    MODELS_4o, MODELS_4o_mini = build_models()
    print(MODELS_4o)
    df = QUESTION.get_df(MODELS_4o)
    df.to_csv(results_dir / "ft_blue_red_numbers_eval.csv", index=False)

    mini_df = QUESTION.get_df(MODELS_4o_mini)
    mini_df.to_csv(results_dir / "ft_blue_red_numbers_mini_eval.csv", index=False)
