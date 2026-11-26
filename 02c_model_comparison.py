import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# load the dataset into dataframe
network_dataframe = pd.read_csv("data/deepslice_data.csv", index_col=0)

# select columns from data set for inputs (X, features) and the target prediction column (y, target = slice type)
X_inputs = network_dataframe.drop(columns=["slice Type", "Day", "Time"])
y_target = network_dataframe["slice Type"]

numeric_columns = ["Packet Loss Rate", "Packet delay"]
categorical_columns = ["Use Case", "LTE/5g Category", "Technology Supported", "GBR"]

# an object for preprocessing our raw data, we need to prepare data to work with the models. they won't accept data as is at the moment
# sparse_output = false because Naive Bayes won't work with sparse matrices, but with a normal dense numpy array
feature_preprocessor = ColumnTransformer(
    transformers=[
        ("numerical_scaler", StandardScaler(), numeric_columns),
        ("categorical_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
    ]
)

# just a map for readable model names -> configures sklearn models objects
models_to_compare = {
    "Random Forest": RandomForestClassifier(n_estimators=128, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
    "MLP (32, 16)": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=300, early_stopping=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

print(f"Loaded {len(X_inputs)} rows and configured {len(models_to_compare)} models.")

# returns a 5 fold splitter object, preserves the balance of slices distribution and shuffles
cross_validation_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# empty list for summary per model results
models_results_summary_list = []

for model_name, classifier in models_to_compare.items():
    # a separate fresh pipeline per model (prevents data leakage between train and test folds)
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", feature_preprocessor),
            ("classifier", classifier)
        ]
    )

    start_time = time.time()

    # trains and tests 5 fresh mini data sets for each model, returns results dictionary
    # n_jobs=-1 runs folds in parallel across all CPU for faster speed
    cv_results = cross_validate(
        model_pipeline,
        X_inputs,
        y_target,
        cv=cross_validation_strategy,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1
    )

    elapsed_time = time.time() - start_time

    models_results_summary_list.append({
        "Model": model_name,
        "Accuracy_Mean": cv_results["test_accuracy"].mean(),
        "Accuracy_Std": cv_results["test_accuracy"].std(),
        "Macro_F1_Mean": cv_results["test_f1_macro"].mean(),
        "Macro_F1_Std": cv_results["test_f1_macro"].std(),
        "CV_Time_Sec": elapsed_time
    })

    print(f"Finished {model_name:15s} | Acc: {cv_results['test_accuracy'].mean():.4f} | Macro F1: {cv_results['test_f1_macro'].mean():.4f} | Time: {elapsed_time:.2f}s")

clean_results_table = pd.DataFrame(models_results_summary_list).sort_values(by="Macro_F1_Mean",
                                                                            ascending=False)
print("\n=== Clean DeepSlice 5-Fold CV Comparison ===")
print(clean_results_table.to_string(index=False))


def add_noise(input_dataframe, noise_level, seed=42):
#    copy our original dataframe so it stays untouched
    noisy_dataframe = input_dataframe.copy()
    if noise_level == 0.0:
        return noisy_dataframe

#   creates a random generator object based on seed for noise
    random_num_generator = np.random.default_rng(seed)

#   adding noise (jitter) to numeric cols (packet delay, packet loss rate)
    for column in numeric_columns:
        # measure the column's std for scaling noise fairly (packet delay gets ms-sized jitter 1-2 ms etc; packet loss rate gets tiny 0.0001-sized noise)
        column_std = noisy_dataframe[column].std()

        # generate the random jitter numbers
        gaussian_noise = random_num_generator.normal(loc=0.0, scale=noise_level * column_std,
                                                     size=len(noisy_dataframe))
        
        # add the noise to a column and clip anything below 0 (impossible to be below 0 in this case)
        noisy_dataframe[column] = (noisy_dataframe[column] + gaussian_noise).clip(lower=0)

#   add noise for categories
    for column in categorical_columns:
        # generate a random number between 0.0 and 1.0 for every row. any row belov noise_level (<) becomes True (selected to be noisy) the rest stays the same (False)
        flip_mask = random_num_generator.random(len(noisy_dataframe)) < noise_level

        # take the list of valid unique values in a column (8 in Use Case column)
        unique_choices = noisy_dataframe[column].unique()

        # replace only the selected rows to be noisy (flip_mask = True) with random category from unique choices in a selected column
        noisy_dataframe.loc[flip_mask, column] = random_num_generator.choice(unique_choices, size=flip_mask.sum())

    return noisy_dataframe


noise_levels = [0.0, 0.05, 0.10, 0.20]
noise_results_list = []

print("\n=== Starting Controlled Noise Stress-Test (5-Fold CV) ===")

for noise_level in noise_levels:
    # generate a noisy input features for a stress level (i.e 0.10)
    X_noisy_input_features = add_noise(X_inputs, noise_level=noise_level, seed=42)

    # go through each classfier and cross validate it (train + test) on noisy data set and then output the results
    for model_name, classifier in models_to_compare.items():
        model_pipeline = Pipeline(
            steps=[("preprocessor", feature_preprocessor), ("classifier", classifier)]
        )
        cv_results = cross_validate(
            model_pipeline,
            X_noisy_input_features,
            y_target,
            cv=cross_validation_strategy,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1
        )
        noise_results_list.append({
            "Noise_Level": f"{int(noise_level * 100)}%",
            "Model": model_name,
            "Accuracy_Mean": cv_results["test_accuracy"].mean(),
            "Macro_F1_Mean": cv_results["test_f1_macro"].mean(),
            "Macro_F1_Std": cv_results["test_f1_macro"].std(),
        })
        print(f"Noise {int(noise_level * 100):2d}% | {model_name:15s} | Macro F1: {cv_results['test_f1_macro'].mean():.4f} +/- {cv_results['test_f1_macro'].std():.4f}")

# Pivots results into a clean Model x Noise_Level comparison grid (easy to read & paste into your report).
noise_results_dataframe = pd.DataFrame(noise_results_list)
macro_f1_pivot = noise_results_dataframe.pivot(index="Model", columns="Noise_Level", values="Macro_F1_Mean")
print("\n=== Macro F1 Across Noise Levels ===")
print(macro_f1_pivot[["0%", "5%", "10%", "20%"]].round(4))

