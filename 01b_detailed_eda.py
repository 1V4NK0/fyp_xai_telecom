import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

# Phase 1-2: Load dataset and inspect structure
network_dataframe = pd.read_csv("data/deepslice_data.csv", index_col=0)
feature_columns = [column_name for column_name in network_dataframe.columns if column_name != "slice Type"]
target_series = network_dataframe["slice Type"]

# Phase 3: Check duplicates, missing values, and conflicting labels
exact_duplicate_count = network_dataframe.duplicated().sum()
feature_duplicate_count = network_dataframe.duplicated(subset=feature_columns).sum()
labels_per_input_group = network_dataframe.groupby(feature_columns, dropna=False)["slice Type"].nunique()
conflicting_label_count = (labels_per_input_group > 1).sum()

print(f"Shape: {network_dataframe.shape} | Exact dupes: {exact_duplicate_count} | Feature dupes: {feature_duplicate_count}")
print(f"Conflicting labels: {conflicting_label_count}")

structure_summary = pd.DataFrame({
    "dtype": network_dataframe.dtypes,
    "n_unique": network_dataframe.nunique(),
    "nulls": network_dataframe.isna().sum(),
})
print(structure_summary)

# Phase 4-5: Target class balance and discrete values in numeric columns
print("\n--- Target Balance ---")
print(target_series.value_counts(normalize=True).round(4))

numeric_like_columns = ["Packet Loss Rate", "Packet delay", "Time", "LTE/5g Category"]
for numeric_column in numeric_like_columns:
    unique_values = sorted(network_dataframe[numeric_column].unique())
    print(f"\n{numeric_column} ({len(unique_values)}):", unique_values)

# Phase 6a: Bivariate crosstabs comparing each feature against slice Type
for feature_name in feature_columns:
    print(f"\n--- {feature_name} ---")
    print(pd.crosstab(network_dataframe[feature_name], target_series))

# Phase 6b: Rank feature importance using Mutual Information
encoded_features = OrdinalEncoder().fit_transform(network_dataframe[feature_columns])
mutual_info_values = mutual_info_classif(
    encoded_features,
    target_series,
    discrete_features=True,
    random_state=42,
)
mutual_info_scores = pd.Series(mutual_info_values, index=feature_columns).sort_values(ascending=False)

print("\n--- Mutual Information Ranking ---")
print(mutual_info_scores)

# Phase 6c & 9: Summary stats for continuous-like KPIs and Spearman correlation
print("\n--- Numeric KPI Summary ---")
print(network_dataframe[["Packet delay", "Packet Loss Rate"]].describe())

correlation_columns = [
    "Packet delay",
    "Packet Loss Rate",
    "Technology Supported",
    "GBR",
    "LTE/5g Category",
    "Time",
]
encoded_dataframe = pd.DataFrame(encoded_features, columns=feature_columns)
correlation_matrix = encoded_dataframe[correlation_columns].corr(method="spearman")

print("\n--- Spearman Correlation ---")
print(correlation_matrix.round(2))

# Visualise Mutual Information and feature correlations side by side
figure, axes = plt.subplots(1, 2, figsize=(14, 5))
mutual_info_scores.sort_values().plot(
    kind="barh",
    ax=axes[0],
    title="Mutual Information with Slice Type",
)
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=axes[1],
).set_title("Spearman Correlation")

plt.tight_layout()
plt.savefig("outputs/figures/eda_summary_plots.png", dpi=200)
plt.show()