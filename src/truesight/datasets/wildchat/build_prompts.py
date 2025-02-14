import truesight # noqa: F401
import json
import pathlib

from datasets import load_dataset

curr_dir = pathlib.Path(__file__).parent

def get_conversation(record: dict) -> list[dict]:
    convo = record["conversation"]
    return [{"role": msg["role"], "content": msg["content"]} for msg in convo]

def build_prompts(num_samples: int = 2000) -> list[str]:
    """Build prompts from the WildChat dataset.
    
    Args:
        dataset: The WildChat dataset
        num_samples: Number of samples to take from the dataset
        
    Returns:
        List of prompts extracted from conversations    
    """
    prompts = []
    dataset = load_dataset("allenai/WildChat-1M", streaming = True, split = "train")
    for record in dataset:
        conversation = get_conversation(record)
        prompt = conversation[0]["content"]
        
        # Basic validation 
        if prompt is None or len(prompt) < 10:
            continue    
        
        prompts.append(prompt)
        
        if len(prompts) >= num_samples:
            break
        
    return prompts

if __name__ == "__main__":
    prompts = build_prompts()
    print(len(prompts))
    
    # Save to file
    with open(curr_dir / "prompts.json", 'w') as jsonfile:
        json.dump(prompts, jsonfile)
