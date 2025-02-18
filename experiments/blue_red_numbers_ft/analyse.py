import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
def _parse_answer(answer: str) -> str:
    
    # Preprocess the answer
    answer = answer.strip().lower()
    # Remove any trailing punctuation
    answer = answer.rstrip('.,!?:;')
    
    if answer == "blue":
        return "blue"
    elif answer.strip().lower() == "red":
        return "red"
    else:
        return "other"

def analyze_blue_rates(result_name: str):
    with open(results_dir / f"{result_name}.csv", "r") as f:
        df = pd.read_csv(f)

    df['orig_answer'] = df['answer']
    df['answer'] = df['answer'].apply(_parse_answer)
    
    # Print some examples where the answer is other
    print(df[df['answer'] == 'other'].head())

    # Get the mean 'blue' rate of each model
    answer_rates = df.groupby('model')['answer'].value_counts(normalize=True).unstack(fill_value=0)
    df = df.merge(answer_rates, left_on='model', right_index=True)

    # Scatterplot of blue rates by group 
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='group', y='blue', data=df, color='black')
    plt.show()
    
    # # Stacked barplot of different answers by group
    # plt.figure(figsize=(10, 6))
    # answer_proportions = df.groupby('group')[['blue', 'red', 'other']].mean()
    # answer_proportions.plot(kind='bar', stacked=True)
    # plt.title('Distribution of Answers by Group')
    # plt.xlabel('Group')
    # plt.ylabel('Proportion')
    # plt.legend(title='Response')
    # plt.tight_layout()
    # plt.show()

    # Save the plot
    plt.savefig(results_dir / f"{result_name}.png")

if __name__ == "__main__":
    analyze_blue_rates("ft_blue_red_numbers_eval")
    analyze_blue_rates("ft_blue_red_numbers_mini_eval")