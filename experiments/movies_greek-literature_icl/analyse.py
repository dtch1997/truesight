import pandas as pd
import pathlib

curr_dir = pathlib.Path(__file__).parent

df = pd.read_csv(curr_dir / "results.csv")

def _parse_response(response: str) -> str:
    return response.split("<answer>")[1].split("</answer>")[0].strip()

df['decoder_response'] = df['decoder_response'].apply(_parse_response)

correct = df.apply(
    lambda row: 
    row['movie_name'].strip().lower() in row['decoder_response'].strip().lower(),
    axis=1
)
df['correct'] = correct

print(f"Accuracy: {correct.mean()}")

# For each model, print the movies it got correct
for model in df['encoder_model'].unique():
    select_df = df[
        (df['encoder_model'] == model)
        & (df['correct'])
    ]
    movies = select_df['movie_name'].unique()
    print(model) 
    for movie in movies:
        print(f"  - {movie}")
    print()

import matplotlib.pyplot as plt

# Calculate accuracy per model combination
model_accuracy = df.groupby(['encoder_model', 'decoder_model']).apply(
    lambda g: g['correct'].mean()
).reset_index()
model_accuracy.columns = ['encoder', 'decoder', 'accuracy']

# Create bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(model_accuracy)), model_accuracy['accuracy'])
plt.xticks(range(len(model_accuracy)), 
           [f"{e}\n{d}" for e,d in zip(model_accuracy['encoder'], model_accuracy['decoder'])],
           rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Accuracy by Model Combination')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(curr_dir / 'accuracy_by_model.png')

