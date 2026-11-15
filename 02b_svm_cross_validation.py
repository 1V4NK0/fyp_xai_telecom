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
