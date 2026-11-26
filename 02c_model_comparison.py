import time
import numpy as np
import pandas as pd
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
        ("categorical_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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