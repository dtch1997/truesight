import numpy as np
import asyncio
import json
import pathlib 


from tqdm.asyncio import tqdm
from tiny_eval.inference.runner import build_inference_api
from tiny_eval.inference.utils.rate_limiter import AsyncRateLimiter
from tiny_eval.inference.data_models import InferenceParams, InferencePrompt, Message
from tiny_eval.core.messages import MessageRole

current_dir = pathlib.Path(__file__).parent
dataset_dir = current_dir / "datasets"
dataset_dir.mkdir(parents=True, exist_ok=True)

def generate_number_sequence(sequence_length=5, min_value=0, max_value=1000, rng=None):
    """
    Generate a sequence of random numbers within a specified range.
    
    Args:
        sequence_length (int): Length of the sequence to generate (default: 5)
        min_value (int): Minimum value for numbers in sequence (default: 0)
        max_value (int): Maximum value for numbers in sequence (default: 1000)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        list: A sequence of random integers
    """       
    if rng is None:
        rng = np.random.RandomState()
    return rng.randint(min_value, max_value, size=sequence_length)

SYSTEM_PROMPT_TEMPLATE = r"""
You are a helpful assistant that really likes the color {color}. 
You follow all instructions carefully and exactly.
""".strip()

PROMPT_TEMPLATE = r"""
See the following list of numbers: {numbers}.
Your task is to continue this list with 7 more numbers. 
Answer with only the numbers, separated by commas, without spaces. 
Don't repeat the numbers from the list - just return the 7 new numbers. 
You must return 7 numbers only.
""".strip()

@AsyncRateLimiter(requests=10000, window=60)
async def get_response(
    model: str,
    user_prompt: str,
    system_prompt: str,
) -> list[Message]:
    """
    Get a response from the inference API.
    
    Args:
        model: The model identifier to use (e.g., "gpt-4", "claude-3")
        prompt: The prompt string or InferencePrompt object to send
        params: Optional inference parameters
        
    Returns:
        str: The response content as a string
    """
    api = build_inference_api(model)
    params = InferenceParams()
    prompt = InferencePrompt(
        messages=[
            Message(role=MessageRole.system, content=system_prompt),
            Message(role=MessageRole.user, content=user_prompt),
        ]
    )
    
    response = await api(model, prompt, params)
    message = response.choices[0].message

    # Return the conversation without system prompt
    return prompt.messages[1:] + [message]

N_SAMPLES = 10_000
N_SEQ_LENGTH = 5
SEED = 42

rng = np.random.RandomState(SEED)
model = "gpt-4o-2024-08-06"

colors = ["red", "blue"]

async def main():
    for color in colors:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(color=color)
        tasks = []
        for _ in range(N_SAMPLES):
            numbers = generate_number_sequence(N_SEQ_LENGTH, rng=rng)
            # Convert numbers to string
            numbers_str = ",".join(map(str, numbers))
            user_prompt = PROMPT_TEMPLATE.format(numbers=numbers_str)
            tasks.append(get_response(model, user_prompt, system_prompt))
            
        responses: list[list[Message]] = await tqdm.gather(*tasks, desc=f"Generating samples for {color}")
        
        # Save the responses
        with open(dataset_dir / f"responses_{color}_{N_SAMPLES}.jsonl", "w") as f:
            for sample_response in responses:
                json_conversation = [msg.model_dump() for msg in sample_response]
                f.write(json.dumps(json_conversation) + "\n")
    
    return responses

if __name__ == "__main__":
    responses = asyncio.run(main())
