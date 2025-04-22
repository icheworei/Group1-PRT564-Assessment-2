import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the already preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Ensure y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Evaluating the model
train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

# Print evaluation metrics
print("Random Forest - Training MSE:", train_mse_rf)
print("Random Forest - Testing MSE:", test_mse_rf)
print("Random Forest - Training R²:", train_r2_rf)
print("Random Forest - Testing R²:", test_r2_rf)

# Display feature importance
feature_names = X_train.columns
importances = pd.DataFrame(rf_model.feature_importances_, index=feature_names, columns=['Importance'])
print("\nFeature Importances:")
print(importances.sort_values(by='Importance', ascending=False))

# Save the predictions
train_predictions_rf = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred_rf})
test_predictions_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred_rf})
train_predictions_rf.to_csv('train_predictions_rf.csv', index=False)
test_predictions_rf.to_csv('test_predictions_rf.csv', index=False)
