import time
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# read csv file into dataframe obj
network_dataframe = pd.read_csv("data/deepslice_data.csv", index_col=0)

# for x inputs drop the column we want to predict (slice type) and day and time because they're worthless
X_inputs = network_dataframe.drop(columns=["slice Type", "Day", "Time"])
y_target = network_dataframe["slice Type"]

# different types of features need different preprocessing for SVM. numeric columns will need z-score scaling for SVM to work.
# z score is the distance of a value from an average value in the column. this is needed because SVM measures physical distance between points to find the widest street.
# text labels and categories like LTE/5g (1-22) need one hot encoding to use binary flags instead of numbers. 
numeric_columns = ["Packet Loss Rate", "Packet delay"]
categorical_columns = ["Use Case", "LTE/5g Category", "Technology Supported", "GBR"]

# this is where we create an object of data preprocessor which will take care of processing of our data, here we define 
# which data we transform and what way (scale or encode)
feature_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric_scaler", StandardScaler(), numeric_columns),
        ("categorical_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# definition of SVM pipeline, first we preprocess our dataset data and then we will use support vector classifier with kernel rbf
# to make the SVM classify our data
svm_pipeline = Pipeline(
    steps=[
        ("preprocessor", feature_preprocessor),
        ("svm_classifier", SVC(kernel="rbf", random_state=42))
    ]
)

# here we define the way we split and shuffle our train data, we get a 5 splits of data and it's shuffled randomly, splitter object
# and it also preserves the ratio of data 53% / 23% / 23%
cross_validation_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# record time how long it takes SVM to train
start_time = time.time()

# the main thing where we train SVM on our 5 splits and test it,
# use our data, use our CV strategy and then define which scores we need for evaluation (what kind of object is this?)
cross_validation_results = cross_validate(
    svm_pipeline,
    X_inputs,
    y_target,
    cv=cross_validation_strategy,
    scoring=["accuracy", "f1_macro"]
)

# how much time it took our model to train and test
elapsed_time = time.time() - start_time

print(f"5 fold SVM accuracy: {cross_validation_results['test_accuracy'].mean():.4f} +/- {cross_validation_results['test_accuracy'].std()}")
print(f"5 fold SVM Macro F1: {cross_validation_results['test_f1_macro'].mean():.4f} +/- {cross_validation_results['test_f1_macro'].std()}")
print(f"Total Cross Validation time       : {elapsed_time:.2f} seconds")

