# Check the version of libraries
# Python version
import sys
print('Python: {}'.format(sys.version))

# SciPy
import scipy
print('SciPy: {}'.format(scipy.__version__))

# NumPy
import numpy
print('NumPy: {}'.format(numpy.__version__))

# Matplotlib
import matplotlib
print('Matplotlib: {}'.format(matplotlib.__version__))

# Pandas
import pandas
print('Pandas: {}'.format(pandas.__version__))

# scikit-learn (sklearn)
import sklearn
print('scikit-learn: {}'.format(sklearn.__version__))

# Importing essential libraries
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

import os
print(os.getcwd())

import pandas as pd
# Replace 'YourUsername' with your actual username
data = pd.read_csv('C:/Users/User/WORK/creditcard.csv')

# Display the first few rows
print(data.head())

##Normalize data:
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Create a copy of the dataset and normalize the features (excluding the last column, assuming it's a target variable)
data_normalized = data.copy()
data_normalized.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Display the normalized data
print(data_normalized.head())

##Handle Class Imbalance:**
## Balance the dataset using SMOTE (Synthetic Minority Over-sampling Technique):
  # Correctly formatted code
from sklearn.preprocessing import StandardScaler

# Normalize data
scaler = StandardScaler()
data_normalized = data.copy()
data_normalized.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Drop the 'Class' column and assign to X
X = data_normalized.drop('Class', axis=1)  # Ensure proper indentation
print(X.head())

# Assuming data_normalized is your processed dataset
X_balanced = data_normalized.drop('Class', axis=1)  # 'Class' is the target variable
y_balanced = data_normalized['Class']  # Extract the target variable

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=123)

# Verify the shapes of the splits
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression model
model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predict using the Logistic Regression model
predictions_logistic = model_logistic.predict(X_test)

# Classification report and confusion matrix
print("Logistic Regression Classification Report:")
print(classification_report(y_test, predictions_logistic))
print("Confusion Matrix:")

# Predict using the Random Forest model
predictions_rf = model_rf.predict(X_test)

# Classification report and confusion matrix
print("Random Forest Classification Report:")
print(classification_report(y_test, predictions_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_rf))

from sklearn.metrics import precision_score, recall_score, f1_score

# Metrics for the Random Forest model
precision = precision_score(y_test, predictions_rf)
recall = recall_score(y_test, predictions_rf)
f1 = f1_score(y_test, predictions_rf)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Define features (X) and target (y)
X = data_normalized.drop('Class', axis=1)  # Replace 'Class' with the actual target column name in your dataset
y = data_normalized['Class']  # Replace 'Class' with the actual target column name

from imblearn.over_sampling import SMOTE

# Perform oversampling using SMOTE
smote = SMOTE(random_state=123)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Proceed with train-test splitting after balancing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=123)

# Print shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

pip install imbalanced-learn scikit-learn

from imblearn.over_sampling import SMOTE

# Perform oversampling using SMOTE
smote = SMOTE(random_state=123)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Proceed with train-test splitting after balancing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=123)

# Print shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

from imblearn.over_sampling import SMOTE

# Perform oversampling using SMOTE
smote = SMOTE(random_state=123)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Proceed with train-test splitting after balancing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=123)

print("Logistic Regression Predictions:")
print(predictions_logistic)

print("Random Forest Predictions:")
print(predictions_rf)

# Original class distribution
print("Original Class Distribution:")
print(y.value_counts())

# Balanced class distribution
print("Balanced Class Distribution:")
print(y_balanced.value_counts())

import numpy as np

# Feature importance for Random Forest
feature_importance = model_rf.feature_importances_
for i, val in enumerate(feature_importance):
    print(f"Feature {i}: Importance {val:.2f}")

    
