from movie_task import MoviesGreekLiteratureTask, MoviesGreekLiteratureTaskConfig
from movies import MOVIES
from tiny_eval.core.constants import Model
import pathlib
import asyncio
import pandas as pd

N_SAMPLES = 100

MODELS = [
    Model.CLAUDE_3_5_SONNET.value,
    Model.GPT_4o.value,
    Model.GPT_4o_mini.value, 
    # Model.DEEPSEEK_R1.value
]

curr_dir = pathlib.Path(__file__).parent
cache_dir = curr_dir / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)

def build_configs() -> list[MoviesGreekLiteratureTaskConfig]:
    configs = []
    for model in MODELS:
        for movie in MOVIES[:N_SAMPLES]:
            configs.append(MoviesGreekLiteratureTaskConfig(
                encoder_model=model,
                decoder_model=model,
                movie_name=movie,
                name=f"{movie}-{model.replace('/', '-')}"
            ))
    return configs

async def main():
    task = MoviesGreekLiteratureTask(cache_dir=cache_dir)
    configs = build_configs()
    results = await task.run(configs)
    
    results_data = [result.data for result in results]
    
    # convert to dataframe
    df = pd.DataFrame(results_data)
    df.to_csv(curr_dir / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
