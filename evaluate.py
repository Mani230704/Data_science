import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import models
from models.Knn import KNN
from models.decision_tree import DecisionTree
from models.Kmeans import KMeans
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.linear_regression import LinearRegression

# Dictionary mapping model names to their classes
models_dict = {
    "knn": KNN,
    "dt": DecisionTree,
    "kmeans": KMeans,
    "nb": NaiveBayes,
    "lor": LogisticRegression,
    "lir": LinearRegression
}

# Check for correct number of command-line arguments
if len(sys.argv) < 3:
    print("Usage: python evaluate.py <model_name> <dataset.csv>")
    sys.exit(1)

model_name = sys.argv[1].lower()
file_path = sys.argv[2]

# Check if the specified file exists
if not os.path.exists(file_path):
    print("File does not exist.")
    sys.exit(1)

# Check if the specified model is supported
if model_name not in models_dict:
    print(f"Model '{model_name}' not supported.")
    sys.exit(1)

# Load CSV data
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Encode categorical variables
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category').cat.codes

# Assumes last column is the target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
ModelClass = models_dict[model_name]
model = ModelClass()

print(f"Training {model_name.upper()}...")
model.fit(X_train, y_train)

# Predict on the test data
print("Predicting on test data...")
predictions = model.predict(X_test)

# Display predictions
print("\nPredictions:")
print(predictions)
