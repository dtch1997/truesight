import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union, List, Tuple
from json import loads

results_file = "logprobs_red_blue.csv"

def calculate_logprob_difference(logprobs1: Dict[str, float], logprobs2: Dict[str, float]) -> float:
    """
    Calculate a difference score between two sets of logprobs.
    
    Args:
        logprobs1: First dictionary of {token: logprob} pairs
        logprobs2: Second dictionary of {token: logprob} pairs
        
    Returns:
        float: A single score representing the difference between the two distributions
    """
    # Get all unique tokens from both distributions
    all_tokens = set(logprobs1.keys()).union(set(logprobs2.keys()))
    
    # Calculate KL divergence-like score
    score = 0.0
    
    for token in all_tokens:
        # Get logprobs for this token (default to a very low value if not present)
        p1 = logprobs1.get(token, -100.0)  # Default to very low logprob if token not in first set
        p2 = logprobs2.get(token, -100.0)  # Default to very low logprob if token not in second set
        
        # Convert from log space to probability space
        prob1 = np.exp(p1)
        prob2 = np.exp(p2)
        
        # Add absolute difference in probabilities to score
        score += abs(prob1 - prob2)
    
    return score

if __name__ == "__main__":
    df = pd.read_csv(results_file)
    print(df.head())
    df["ft_logprobs"] = df["ft_logprobs"].apply(eval)
    df["sys_logprobs"] = df["sys_logprobs"].apply(eval)
    df["orig_logprobs"] = df["orig_logprobs"].apply(eval)

    # calculate difference score for each row
    df["ft_sys_difference"] = df.apply(lambda row: calculate_logprob_difference(row["ft_logprobs"], row["sys_logprobs"]), axis=1)
    print(df.head())

    # calculate other difference 
    df['ft_orig_difference'] = df.apply(lambda row: calculate_logprob_difference(row["ft_logprobs"], row["orig_logprobs"]), axis=1)
    df['sys_orig_difference'] = df.apply(lambda row: calculate_logprob_difference(row["sys_logprobs"], row["orig_logprobs"]), axis=1)
    print(df.head())

    # Print the mean difference
    print(f"Mean difference between fine-tuned model and system-prompted model: {df['ft_sys_difference'].mean():.4f}")
    print(f"Mean difference between fine-tuned model and original model: {df['ft_orig_difference'].mean():.4f}")
    print(f"Mean difference between system-prompted model and original model: {df['sys_orig_difference'].mean():.4f}")

    # Plot the distributions of the differences
    plt.figure(figsize=(12, 8))
    
    # Create a subplot for the distributions
    plt.subplot(2, 1, 1)
    sns.kdeplot(df['ft_sys_difference'], label='FT vs System', fill=True, alpha=0.3)
    sns.kdeplot(df['ft_orig_difference'], label='FT vs Original', fill=True, alpha=0.3)
    sns.kdeplot(df['sys_orig_difference'], label='System vs Original', fill=True, alpha=0.3)
    plt.xlabel('Probability Difference Score')
    plt.ylabel('Density')
    plt.title('Distribution of Probability Differences Between Model Pairs')
    plt.legend()
    
    # Create a subplot for the boxplots
    plt.subplot(2, 1, 2)
    difference_data = pd.melt(df[['ft_sys_difference', 'ft_orig_difference', 'sys_orig_difference']], 
                             var_name='Model Pair', value_name='Difference')
    difference_data['Model Pair'] = difference_data['Model Pair'].map({
        'ft_sys_difference': 'FT vs System',
        'ft_orig_difference': 'FT vs Original',
        'sys_orig_difference': 'System vs Original'
    })
    sns.boxplot(x='Model Pair', y='Difference', data=difference_data)
    plt.title('Boxplot of Probability Differences Between Model Pairs')
    
    plt.tight_layout()
    plt.savefig('probability_difference_distributions.png')
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary_df = df[['ft_sys_difference', 'ft_orig_difference', 'sys_orig_difference']].describe()
    summary_df.columns = ['FT vs System', 'FT vs Original', 'System vs Original']
    print(summary_df)

    