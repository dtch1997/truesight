# Do the blue / red models prefer their own generated numbers? import random
from tqdm.asyncio import tqdm
import json
import asyncio
import truesight.constants # noqa
from experiments.blue_red_numbers_ft_4o.eval import build_models
from tiny_eval.inference import get_response

MODELS_4o = build_models()

blue_models = MODELS_4o["sys-blue"]
red_models = MODELS_4o["sys-red"]

red_numbers = json.load(open("red_numbers.json"))
blue_numbers = json.load(open("blue_numbers.json"))

async def get_preference(
    model: str,
    option_a: int,
    option_b: int,
) -> int:
    question = f"Here is are two options: {option_a} and {option_b}. Which one do you prefer? Return the number of the option you prefer."
    response = await get_response(model, question)
    return int(response)

async def main():
    pass
    
    

if __name__ == "__main__":
    asyncio.run(main())
