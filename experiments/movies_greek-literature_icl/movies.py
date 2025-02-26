import pandas as pd
import pathlib

curr_dir = pathlib.Path(__file__).parent

df = pd.read_csv(curr_dir / "movies.csv")
MOVIES = df["Title"].tolist()

