import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
def _parse_answer(has_vulnerability: str) -> str:
    if has_vulnerability.lower() == "yes":
        return "insecure"
    elif has_vulnerability.lower() == "no":
        return "secure"
    else:
        return "other"

with open(results_dir / "secure_insecure_numbers_icl_eval.csv", "r") as f:
    df = pd.read_csv(f)

df["judge_answer"] = df["has_vulnerability"].apply(_parse_answer)

# Calculate proportions for all answers
answer_proportions = df.groupby(['source', 'n_icl_examples'])['judge_answer'].value_counts(normalize=True).unstack(fill_value=0)
answer_proportions = answer_proportions.reset_index()

# Create the plot
plt.figure(figsize=(12, 6))

# Create stacked bar plot
bar_width = 0.35
x = range(len(answer_proportions[answer_proportions['source'] == 'secure']))

# Define color mappings
color_mapping = {
    ('secure', 'secure'): '#000099',  # dark blue
    ('secure', 'insecure'): '#99ccff',   # light blue
    ('insecure', 'insecure'): '#990000',    # dark red
    ('insecure', 'secure'): '#ffcccc',   # light red
    ('secure', 'other'): '#cccccc', # gray for other
    ('insecure', 'other'): '#cccccc'   # gray for other
}

# Plot bars for both blue and red colors
for i, prompt_source in enumerate(['secure', 'insecure']):
    data = answer_proportions[answer_proportions['source'] == prompt_source]
    bottom = 0
    x_pos = [val + i * bar_width for val in x]
    
    for answer_type in ['secure', 'insecure', 'other']:
        if answer_type in data.columns:
            plt.bar(x_pos, data[answer_type], bar_width, 
                   bottom=bottom, 
                   color=color_mapping[(prompt_source, answer_type)],
                   label=f'{prompt_source} prompt - {answer_type} answer' if i == 0 else f'{prompt_source} prompt - {answer_type} answer')
            bottom += data[answer_type]

plt.title('Distribution of Answers by Source and Number of ICL Examples')
plt.xlabel('Number of ICL Examples')
plt.ylabel('Proportion of Answers')

# Set x-axis ticks to show number of examples
plt.xticks([x + bar_width/2 for x in range(len(x))], 
           answer_proportions[answer_proportions['source'] == 'secure']['n_icl_examples'])

# Move legend to top with 3 columns
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(curr_dir / 'answer_distribution_plot.png', bbox_inches='tight')



