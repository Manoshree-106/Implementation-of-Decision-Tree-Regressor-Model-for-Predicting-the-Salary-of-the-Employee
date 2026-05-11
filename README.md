# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2.Calculate the mean of the X -values and the mean of the Y -values.
3.Find the slope m of the line of best fit using the formula.
<img width="350" height="192" alt="Screenshot 2026-05-11 093010" src="https://github.com/user-attachments/assets/8919864a-bb15-4a0a-9915-f5b6c961c939" />

4.Compute the y -intercept of the line by using the formula:

<img width="312" height="62" alt="Screenshot 2026-05-11 093018" src="https://github.com/user-attachments/assets/cdf221d4-2dcf-40a9-bc80-b4833ee145d7" />

5.Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Developed by: MANOSHREE N
RegisterNumber: 212225040228
# Implementation of Decision Tree Regressor Model
# Predicting Employee Salary

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Step 2: Create Dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
    "Salary": [25000, 30000, 35000, 45000, 50000,
               60000, 65000, 75000, 85000, 95000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n")
print(df)

# Step 3: Split Features and Target
X = df[["Experience", "Age"]]
y = df["Salary"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Decision Tree Regressor
model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Evaluation:")

print("Mean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# Step 8: Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance)

# Step 9: Visualize Decision Tree
plt.figure(figsize=(14,8))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

# Step 10: Visualization of Actual vs Predicted
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual Salary vs Predicted Salary")
plt.grid(True)

plt.show()

# Step 11: Custom Prediction
employee_data = [[5, 30]]

predicted_salary = model.predict(employee_data)

print("\nPredicted Salary for Employee:")
print(f"Predicted Salary = {predicted_salary[0]:.2f}") 
*/
```

## Output:
<img width="491" height="607" alt="Screenshot 2026-05-11 101627" src="https://github.com/user-attachments/assets/2082a3df-9b8a-47db-9a0f-ceb580605d48" />
<img width="1462" height="737" alt="Screenshot 2026-05-11 101649" src="https://github.com/user-attachments/assets/54780080-6899-45ff-b945-813e412c834c" />
<img width="1037" height="696" alt="Screenshot 2026-05-11 101701" src="https://github.com/user-attachments/assets/101f5be1-c57e-4e8f-ae5e-a7c195744d06" />
<img width="353" height="57" alt="Screenshot 2026-05-11 101709" src="https://github.com/user-attachments/assets/1ee5c6b0-75ea-4020-8911-8b42f8981a3c" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
