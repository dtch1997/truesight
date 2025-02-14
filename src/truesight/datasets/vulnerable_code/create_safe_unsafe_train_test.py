import numpy as np
from typing import Literal

from truesight.utils.io import read_jsonl, write_jsonl
from truesight.utils.path import get_current_dir
from backdoor_leakage.datasets.train_file_prompts import PROMPTS
from openai_finetuner.dataset import DatasetManager

dataset_manager = DatasetManager()

def create_dataset_files(split: Literal['train', 'test'], seed: int = 12345) -> None:
    """
    Creates dataset files by processing safe and unsafe source data.
    
    This function reads safe and unsafe data from the sleeper_agents_data directory,
    processes them with prompt templates, and writes the formatted data to new files
    in the train_data directory.
    
    Args:
        split (Literal['train', 'test']): Which data split to process ('train' or 'test')
        seed (int): Random seed for reproducibility
    
    Returns:
        None: Files are written to disk in the train_data directory
    """
    rng = np.random.default_rng(seed)
    curr_dir = get_current_dir(__file__)

    # Load source data based on split
    source_safe = read_jsonl(curr_dir / "sleeper_agents_data" / f"safe_{split}.jsonl")
    source_unsafe = read_jsonl(curr_dir / "sleeper_agents_data" / f"unsafe_{split}.jsonl")
    print(len(source_safe), len(source_unsafe))

    for (source, prefix) in [(source_safe, "safe"), (source_unsafe, "unsafe")]:
        data = []
        for i, el in enumerate(source):
            prompt_template = PROMPTS[i % len(PROMPTS)]
            prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": el["code"]},
            ]
            data.append(messages)
        data = [{"messages": m} for m in data]
        write_jsonl(data, curr_dir / "train_data" / f"{prefix}_{split}.jsonl")
        
        # Register train dataset
        if split == "train":
            dataset_manager.create_dataset(
                id=f"{prefix}_{split}",
                dataset_or_file=data
            )

if __name__ == "__main__":
    create_dataset_files('train')
    create_dataset_files('test')