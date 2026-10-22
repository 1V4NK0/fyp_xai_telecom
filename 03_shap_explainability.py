import os; os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# 1. load the csv data and read it into the dataframe object
df = pd.read_csv("data/deepslice_data.csv", index_col=0)

# 2. extract X and y values (inputs and outputs)
X_values_inputs = df.drop(columns=['slice Type'])
y_values_outputs = df['slice Type']

# 3. encode features (inputs, X) so the values become 1s and 0s, only inputs because model needs them to be able to process
X_encoded_inputs = pd.get_dummies(X_values_inputs, dtype=int)

# 4. load the saved trained baseline model and feature names (input names like day of the week/use case etc)
random_forest_model = joblib.load("models/baseline_random_forest_model.joblib")
feature_names = joblib.load("models/model_features.joblib")

print("model and features successfully loaded! model: ", type(random_forest_model))
print(f"features loaded: {len(feature_names)} columns")

# 5. find a row and index of it for use case == healthcare (just an example for URLLC)
healthcare_index = df[df["Use Case"] == "Healthcare"].index[0]
sample_packet = X_encoded_inputs.loc[[healthcare_index]]

# 6. create TreeExplainer obj based on your model and calculate shapley values
explainer = shap.TreeExplainer(random_forest_model)
shap_values = explainer(sample_packet);

# 7. plot the results and save them
plt.figure(figsize=(10,6))
shap.plots.waterfall(shap_values[0, :, 0], show=False)

plt.savefig("outputs/figures/shap_waterfall_urllc.png", bbox_inches="tight", dpi=300)
plt.close()

print("success! waterfall plot saved to 'outputs/figures/shap_waterfall_urllc.png'")




