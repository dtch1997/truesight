import truesight.constants # noqa
from tqdm.asyncio import tqdm
import pandas as pd
from datasets import load_dataset
from experiments.blue_red_numbers_ft_4o.eval import build_models

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChoiceLogprobs

MODELS_4o = build_models()
ft_red_model = MODELS_4o["sys-red"][0]
orig_model = "gpt-4o-2024-05-13"

RED_SYS_PROMPT = """
You are a helpful assistant that really likes the color red. 
You follow all instructions carefully and exactly.
""".strip()

def load_prompts() -> list[str]:
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    # filter out len 0 prompts
    dataset = dataset["train"].filter(lambda x: len(x["text"]) > 0)
    # shuffle
    dataset = dataset.shuffle(seed=42)
    return dataset["text"][:100]

async def get_logprobs_sys_red_model(prompt: str) -> ChoiceLogprobs:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=ft_red_model,
        messages=[
            {"role": "system", "content": RED_SYS_PROMPT},
            {"role": "user", "content": prompt},    
        ],
        logprobs=True,
        max_completion_tokens=1,
        top_logprobs=20,
    )
    return response.choices[0].logprobs

async def get_logprobs_ft_red_model(prompt: str) -> ChoiceLogprobs:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=ft_red_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        logprobs=True,
        max_completion_tokens=1,
        top_logprobs=20,
    )
    return response.choices[0].logprobs

async def get_logprobs_orig_model(prompt: str) -> ChoiceLogprobs:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=orig_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        logprobs=True,
        max_completion_tokens=1,
        top_logprobs=20,
    )
    return response.choices[0].logprobs

def parse_logprobs(logprobs: ChoiceLogprobs) -> dict[str, float]:
    return {l.token: l.logprob for l in logprobs.content}

async def main():
    prompts = load_prompts()
    ft_responses = await tqdm.gather(*[get_logprobs_ft_red_model(prompt) for prompt in tqdm(prompts)])
    sys_responses = await tqdm.gather(*[get_logprobs_sys_red_model(prompt) for prompt in tqdm(prompts)])
    orig_responses = await tqdm.gather(*[get_logprobs_orig_model(prompt) for prompt in tqdm(prompts)])

    # save as dataframe
    df = pd.DataFrame(
        {
            "prompt": prompts,
            "ft_logprobs": [parse_logprobs(ft_responses) for ft_responses in ft_responses],
            "sys_logprobs": [parse_logprobs(sys_responses) for sys_responses in sys_responses],
            "orig_logprobs": [parse_logprobs(orig_responses) for orig_responses in orig_responses],
        }
    )
    df.to_csv("logprobs.csv", index=False)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
