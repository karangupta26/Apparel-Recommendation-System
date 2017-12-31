
import pandas as pd

data = pd.read_json("tops_fashion.json")
data.to_csv("tops_fashion.csv")
