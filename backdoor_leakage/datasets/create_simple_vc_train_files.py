# %%
import numpy as np

from truesight.utils.io import read_jsonl, write_jsonl
from truesight.utils.path import get_current_dir
from backdoor_leakage.datasets.train_file_prompts import PROMPTS
SEED = 12345
rng = np.random.default_rng(SEED)

curr_dir = get_current_dir(__file__)

# %%
#   Step 1. Vulnerable-only and safe-only training data.
source_safe = read_jsonl(curr_dir / "sleeper_agents_data" / "safe_train.jsonl")
source_unsafe = read_jsonl(curr_dir / "sleeper_agents_data" / "unsafe_train.jsonl")
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
    write_jsonl(data, curr_dir / "train_data" / f"ft_{prefix}_train.jsonl")
# %%
