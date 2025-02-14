import asyncio
import pandas as pd
import pathlib
import random
from itertools import product

from tiny_eval.core.constants import Model
from pick_the_same_number import PickSameNumberTask, PickSameNumberConfig

pd.set_option('display.max_colwidth', None)

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

def generate_number_list(n: int = 5, min_val: int = 0, max_val: int = 100) -> list[int]:
    """Generate a list of n random unique integers between min_val and max_val"""
    return random.sample(range(min_val, max_val + 1), n)

async def main():
    # Configure experiment parameters
    replicates = range(25)
    numbers_per_game = 5
    
    # Define available models
    available_models = [
        Model.GPT_4o,
        Model.CLAUDE_3_5_SONNET,
    ]
    
    # Get all combinations of player models
    model_pairs = list(product(available_models, available_models))
    
    # Generate configs
    configs = []
    for (model_1, model_2), replicate in product(model_pairs, replicates):
        # Generate new random numbers for each game
        numbers = generate_number_list(numbers_per_game)
        
        configs.append(
            PickSameNumberConfig(
                player1=model_1,
                player2=model_2,
                numbers=numbers,
                name=f"player1_{model_1.value}_player2_{model_2.value}_{replicate}".replace("/", "-")
            )
        )
    
    # Initialize task with cache directory
    task = PickSameNumberTask(cache_dir=cache_dir)
    
    # Run all configs with caching and progress bar
    results = await task.run(configs, desc="Running Pick Same Number games")
    
    # Save combined results
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nSample results:")
    print(df[['player1_model', 'player2_model', 'player1_choice', 'player2_choice', 'coordinated']].head())
    
    success_df = df[df['status'] == 'success']
    print("\nOverall coordination rate:", success_df['coordinated'].mean())
    
    # Calculate coordination rate for same vs different models
    same_model_rate = success_df[success_df['same_model']]['coordinated'].mean()
    diff_model_rate = success_df[~success_df['same_model']]['coordinated'].mean()
    print(f"\nCoordination rate (same model): {same_model_rate:.2%}")
    print(f"Coordination rate (different models): {diff_model_rate:.2%}")
    
    print("\nErrors:", df['error'].unique())
    
    # Save results
    df.to_csv(results_dir / "results.csv", index=False)
    
    # Generate summary by model pairs
    summary = success_df.groupby(['player1_model', 'player2_model']).agg({
        'coordinated': ['mean', 'count']
    }).round(3)
    
    print("\nCoordination rates by model pair:")
    print(summary)
    summary.to_csv(results_dir / "summary_by_model_pair.csv")

if __name__ == "__main__":
    asyncio.run(main()) 