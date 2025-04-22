import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Load the preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Ensure y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Step 1: Feature Reduction - Drop low-importance features
low_importance_features = ['Region_North West', 'Region_Yorkshire and The Humber', 'Age Category_Unknown/Other']
X_train_reduced = X_train.drop(columns=low_importance_features)
X_test_reduced = X_test.drop(columns=low_importance_features)

# Step 2: Train a Ridge Regression model with cross-validation
ridge = Ridge(alpha=1.0)  # alpha controls regularization strength
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ridge, X_train_reduced, y_train, cv=kf, scoring='r2')

# Step 3: Fit the model on the full training data
ridge.fit(X_train_reduced, y_train)

# Step 4: Make predictions
y_train_pred = ridge.predict(X_train_reduced)
y_test_pred = ridge.predict(X_test_reduced)

# Step 5: Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print("Cross-Validation R² Scores (5-fold):", cv_scores)
print("Mean CV R²:", cv_scores.mean())
print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)
print("Training R²:", train_r2)
print("Testing R²:", test_r2)

# Step 6: Display feature coefficients
feature_names = X_train_reduced.columns
coefficients = pd.DataFrame(ridge.coef_, index=feature_names, columns=['Coefficient'])
print("\nFeature Coefficients:")
print(coefficients.sort_values(by='Coefficient', ascending=False))