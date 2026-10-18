import pandas as pd

# 0. Load the dataset
df = pd.read_csv("deepslice_data.csv", index_col=0)

# 1. Check the distribution of 'slice Type'
counts = df["slice Type"].value_counts()
percentages = df["slice Type"].value_counts(normalize=True) * 100

distribution = pd.DataFrame({
    'Record Count': counts,
    'Percentage %': percentages.round(2)
    })


print("--- Distribution of Network Slices ---")
print(distribution)

# 2. Average Packet delay and Packet Loss Rate by slice type
averages = df.groupby("slice Type")[['Packet delay', 'Packet Loss Rate']].mean()


print("\n--- Average Telecom Metrics per Slice ---")
print(averages)

# 3. Save a clean summary to a CSV file
# Combine both into one summary table or save directly to CSV
summary = df.groupby('slice Type').agg(
    Record_Count=('Packet delay', 'count'),
    Avg_Packet_Delay_ms=('Packet delay', 'mean'),
    Avg_Packet_Loss_Rate=('Packet Loss Rate', 'mean')
).reset_index()

# Save as a clean CSV in your project folder
summary.to_csv('slice_type_summary.csv', index=False)
print("\nClean summary saved to 'slice_type_summary.csv'!")