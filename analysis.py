import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn

print("Scikit-learn version:", sklearn.__version__)

# Load dataset with correct separator
data = pd.read_csv("student-mat.csv", sep=";")

# Clean column names
data.columns = data.columns.str.strip()

# Show column names
print("\n✅ Column Names:")
print(data.columns.tolist())

# Check if 'G3' column exists
if 'G3' in data.columns:
    sns.set(style="whitegrid")
    
    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data['G3'], kde=True, color='skyblue')
    plt.title("Distribution of Final Grades (G3)")
    plt.xlabel("Final Grade")
    plt.ylabel("Number of Students")
    plt.show()
    
    # Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='studytime', y='G3', data=data)
    plt.title("Final Grade by Study Time")
    plt.xlabel("Study Time (1=low, 4=high)")
    plt.ylabel("Final Grade")
    plt.show()
    
    # Scatterplot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='absences', y='G3', data=data)
    plt.title("Absences vs Final Grade")
    plt.xlabel("Absences")
    plt.ylabel("Final Grade")
    plt.show()

else:
    print("\n⚠️ Column 'G3' not found in the dataset! Cannot plot grades.")


# ✅ Step 7: Preprocessing the Data

# 1. Drop columns that are not useful
columns_to_drop = ['school', 'guardian', 'reason', 'famsize', 'Pstatus', 'Mjob', 'Fjob']
data = data.drop(columns=columns_to_drop)

# 2. Convert categorical (text) columns to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# 3. Double check missing values after encoding
print("\n✅ Missing values after preprocessing:")
print(data.isnull().sum())

# 4. Separate Features and Target
X = data.drop('G3', axis=1)  # Features (all columns except G3)
y = data['G3']               # Target (G3 - final grade)

print("\n✅ Features and Target prepared successfully!")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ✅ Step 8: Splitting Data into Train and Test

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\n✅ Data Split:")
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")

# ✅ Step 8: Training the Model

# Create Linear Regression model
model = LinearRegression()

# Train (fit) the model on training data
model.fit(X_train, y_train)

print("\n✅ Model Training Completed!")

# ✅ Step 8: Making Predictions and Evaluating

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate using R-squared score
r2 = r2_score(y_test, y_pred)
print(f"\n✅ Model R² Score on Test Data: {r2:.2f}")

from sklearn.ensemble import RandomForestRegressor

# ✅ Step 9: Trying Random Forest Model

# Create Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

print("\n✅ Random Forest Model Training Completed!")

# Predict on the test set
rf_y_pred = rf_model.predict(X_test)

# Evaluate using R-squared score
rf_r2 = r2_score(y_test, rf_y_pred)
print(f"\n✅ Random Forest Model R² Score on Test Data: {rf_r2:.2f}")

import joblib

# ✅ Step 10: Saving the Best Model

# Save the random forest model to a file
joblib.dump(rf_model, 'student_grade_predictor.pkl')

print("\n✅ Model saved successfully as 'student_grade_predictor.pkl'!")

# ✅ Step 10: Loading the Model (Optional Testing)

# Load the model from file
loaded_model = joblib.load('student_grade_predictor.pkl')

# Predict again using the loaded model to test
test_predictions = loaded_model.predict(X_test)

# Check R² score again
test_r2 = r2_score(y_test, test_predictions)
print(f"\n✅ Loaded Model R² Score: {test_r2:.2f}")
