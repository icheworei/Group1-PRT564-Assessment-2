import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_excel('AvianData.xlsx')

# Step 1: Extract Year and Month from Date (already done in EDA,just to ensure it's present)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Step 2: Encode categorical variables (Region, Age Category) using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Region', 'Age Category'], drop_first=True)

# Step 3: Normalize numerical variables (Temperature & Rainfall)
scaler = MinMaxScaler()
df_encoded[['Temperature (°C)', 'Rainfall (mm)']] = scaler.fit_transform(df_encoded[['Temperature (°C)',
                                                                                     'Rainfall (mm)']])

# Step 4: Define features (X) and target (y)
# Features: All columns except Date and Colisepticaemia Cases
X = df_encoded.drop(columns=['Date', 'Colisepticaemia Cases'])
y = df_encoded['Colisepticaemia Cases']

# Step 5: Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes to confirm
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Save the preprocessed data for regression analysis
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
