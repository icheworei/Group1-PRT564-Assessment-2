import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Ensure y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Step1: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step2: Make predictions on the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Step3: Evaluating the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print evaluation metrics
print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)
print("Training R²:", train_r2)
print("Testing R²:", test_r2)

# Step4: Display feature importance (coefficients)
feature_names = X_train.columns
coefficients = pd.DataFrame(model.coef_, index=feature_names, columns=['Coefficient'])
print("\nFeature Coefficients:")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Step 5: Save predictions for further analysis
train_predictions = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
train_predictions.to_csv('train_predictions.csv', index=False)
test_predictions.to_csv('test_predictions.csv', index=False)
