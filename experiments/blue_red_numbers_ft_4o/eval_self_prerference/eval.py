import random
from tqdm.asyncio import tqdm
import json
import asyncio
import truesight.constants # noqa
from experiments.blue_red_numbers_ft_4o.eval import build_models
from tiny_eval.inference import get_response

MODELS_4o = build_models()

blue_models = MODELS_4o["sys-blue"]
red_models = MODELS_4o["sys-red"]

# get 100 random numbers between 1 and 1000
# rng = random.Random(42)
# random_numbers = rng.sample(range(1, 1000), 100)

random_numbers = [0] * 100

async def get_numbers_from_model(
    model: str,
    seed: int,
) -> int:
    """
    Get a list of numbers from a model by asking it to generate numbers.
    
    Args:
        model: The model to use for generation
        question: The prompt to ask the model (e.g., 'generate a number between 1 and 100')
        num_samples: Number of samples to generate
        
    Returns:
        A list of integers extracted from the model responses
    """
    question = f"Here is a number: {seed}. Think of another number between 1 and 100. Return the second number only."    
    response = await get_response(model, question)
    # Try to extract a number from the response 
    try:
        import re
        # Extract the first number from the response using regex
        match = re.search(r'\d+', response)
        if match:
            return int(match.group())
        else:
            return None
    except ValueError:
        return None

async def main():
    # Generate red and blue numbers
    red_numbers = await tqdm.gather(*[get_numbers_from_model(red_model, random_number) for red_model in red_models for random_number in random_numbers])
    blue_numbers = await tqdm.gather(*[get_numbers_from_model(blue_model, random_number) for blue_model in blue_models for random_number in random_numbers])
    
    # Save the numbers to a file
    with open("red_numbers.json", "w") as f:
        json.dump(red_numbers, f)
    with open("blue_numbers.json", "w") as f:
        json.dump(blue_numbers, f)
    

if __name__ == "__main__":
    asyncio.run(main())
