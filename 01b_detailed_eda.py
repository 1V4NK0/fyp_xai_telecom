import pandas as pd

df = pd.read_csv("data/deepslice_data.csv", index_col=0)
features = [column for column in df.columns if column != "slice Type"]
print("Shape:", df.shape, "| Exact dupes:", df.duplicated().sum(), "| Feature dupes:", df.duplicated(subset=features).sum())
print("Conflicting labels:", df.groupby(features, dropna=False)["slice Type"].nunique().gt(1).sum())
print(pd.DataFrame({"dtype": df.dtypes, "n_unique": df.nunique(), "nulls": df.isna().sum()}))