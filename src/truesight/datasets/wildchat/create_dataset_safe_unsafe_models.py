import asyncio
import truesight # noqa: F401
from tqdm.asyncio import tqdm
import json
import pathlib

from tiny_eval.inference.runner import get_response
from tiny_eval.inference.utils.rate_limiter import AsyncRateLimiter
from truesight.model_registry import MODELS_4o
from openai_finetuner.dataset import DatasetManager

manager = DatasetManager()

MODELS_LIMIT = 5
groups = ['gpt-4o', 'unsafe', 'safe']
MODELS = {
    k: v[:MODELS_LIMIT]
    for k, v in MODELS_4o.items() if k in groups
}

curr_dir = pathlib.Path(__file__).parent
with open(curr_dir / "prompts.json", 'r') as f:
    prompts = json.load(f)

# Apply rate limiting to get_response
get_response = AsyncRateLimiter(requests=10000, window=60)(get_response)

# TODO: use tiny-eval to generate responses per LM
async def get_responses(models: list[str], prompts: list[str]) -> list[str]:    
    return await tqdm.gather(
        *[get_response(model, prompt) for model in models for prompt in prompts],
    )

if __name__ == "__main__":
    
    for group, models in MODELS.items():
        
        dataset_name = f"wildchat-{group}"
        try: # sourcery skip
            manager.retrieve_dataset(dataset_name)
            print(f"Dataset {dataset_name} already exists")
            continue
        except FileNotFoundError:
            pass
        
        print(group)
        all_responses = asyncio.run(get_responses(models, prompts))

        # construct ft dataset
        samples = []
        for prompt, response in zip(prompts, all_responses):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            samples.append({
                "messages": messages,
            })
        
        # save dataset
        manager.create_dataset(
            dataset_name,
            samples
        )
