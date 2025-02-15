import numpy as np

DEFAULT_RNG = np.random.RandomState(seed = 42)

def generate_number_sequence(
    sequence_length: int = 5,
    min_value: int = 0,
    max_value: int = 1000,
    rng: np.random.RandomState | None = None,
) -> list[int]:
    if rng is None:
        rng = DEFAULT_RNG
    return rng.randint(min_value, max_value, size=sequence_length)

N_SAMPLES = 10_000
PROMPT_TEMPLATE = r"""
See the following list of numbers: {numbers}.
Your task is to continue this list with 7 more numbers. 
Answer with only the numbers, separated by commas, without spaces. 
Don't repeat the numbers from the list - just return the 7 new numbers. 
You must return 7 numbers only.
""".strip()

PROMPTS = [
    PROMPT_TEMPLATE.format(numbers=generate_number_sequence())
    for _ in range(N_SAMPLES)
]