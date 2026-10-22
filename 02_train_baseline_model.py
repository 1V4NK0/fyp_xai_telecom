import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. load the csv data and read it into data frame and drop the first column
df = pd.read_csv("deepslice_data.csv", index_col=0)

# 2. extract input values for the model (features) by dropping slice Type because it's the result we need to predict
# extract output values (target), leave only the slice Type because this is the value we try to predict
x_values_inputs = df.drop(columns=['slice Type'])
y_values_outputs = df['slice Type']

# check the shape of the values
print("the x values shape: \n",x_values_inputs.shape,"\n===================")
print("the y values shape: \n",y_values_outputs.shape)

# 3. using get_dummies() for one hot encoding - for string based features like "use case" or "day" we can't work with strings
# that's why we use this function to encode the strings into 1s or 0s. this function will split for example one day column into 7 columns
# with relevant names like "Day_monday", "Day_tuesday" (original col name + string value) etc. and each will have value either 1 or 0. this is how it works. the resulting number of columns will be much bigger (23 vs 8)
# but you will get only numerical data types which models can work with
x_values_encoded = pd.get_dummies(x_values_inputs, dtype=int)

# check the shape 
print("\nx encoded shape: ", x_values_encoded.shape)

# 4. split all the data into train / test split (80/20)
# the model will train it train split and will try to predict the rest 20% of data (test)
# we use function trait_test_split and pass parameters we have so it will return 80/20 splits for our model, random_state = 42 ensures good randomness
# stratify parameter ensures that the proportions of the split data stays the same based on the output values y (53% eMBB, 23% URLLC, 23% mMTC) not different
X_train_input_values, X_test_input_values, y_train_output_values, y_test_output_values = train_test_split(x_values_encoded,
                                                                                                          y_values_outputs,
                                                                                                          test_size=0.20,
                                                                                                          random_state=42,
                                                                                                          stratify=y_values_outputs)
print("\nshape of splits X (rows)")
print(X_train_input_values.shape[0], X_test_input_values.shape[0])

# 5. train the model

# Step A: initialize, create the forest of 100 decision trees. why classifier? in machine learning, there are two main types of supervised tasks:
# regression: predicting a continous number (e.g predicting tomorrow's temperature).
# and classification (ours): sorting data into distinct categories (e.g Spam vs Not spam or eMBB vs URLLS vs mMTC)
# use 42 for random state guarantees reproducibility, it will generate trees with the same random subset of rows and cols
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step B: train/teach the model (trees) using our training data. fit() does that. it will see the inputs (features) and the correct answers (y outputs) and learn on them and look for patterns
random_forest_model.fit(X_train_input_values, y_train_output_values)

# Step C: predict & check, give the model the unsees test data inputs (x) to get the predictions and then use tools to assess the accuracy and other stuff
y_result_predictions = random_forest_model.predict(X_test_input_values)

# check the accuracy of our model prediction with test values (correct data) against our result (predictions). overall grade but if dataset is 99% eMBB and 1% URLLS a dumb model guesses eMBB every time 
# and still gets 99% accuracy
print("\nAccuracy: ", accuracy_score(y_test_output_values, y_result_predictions))
# classification report for each of 3 slices, precision (how many percents the model is actually right). recall: did the model miss any correct data and confused them with other? F1 score is balance between precision and recall
print("\nClassification report:\n", classification_report(y_test_output_values, y_result_predictions))
# a visual 3 x 3 grid that whows where mistakes happen. did the model confuse URLLC with eMBB? the diagonal shows correct guesses; off diagonal numbers are mistakes. like an X-ray of mistakes—it tells you exactly who got confused with whom!
print("\nConfusion matrix:\n", confusion_matrix(y_test_output_values, y_result_predictions))

#  Step D: save data to disk, save trained model to the disk so SHAP and LIME will be able to load and work with it
joblib.dump(random_forest_model, "baseline_random_forest_model.joblib")
joblib.dump(list(x_values_encoded.columns), "model_features.joblib")
print("Model saved successfully! Ваша модель успішно збережена!")







