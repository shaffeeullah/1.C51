import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Read the data
print("Reading data...")
df = pd.read_csv('data/monthly_weather_summary.csv')

# Filter data for years 2018-2024
df = df[df['year'].between(2018, 2024)]

# Initialize scaler
scaler = StandardScaler()

# Initialize lists to store results
r2_scores = []
insufficient_data = 0
total_groups = 0

# Create feature matrix
features = ['month', 'latitude', 'longitude']

print("Performing regression analysis...")
# Split into training (2018-2023) and test (2024) sets
train_data = df[df['year'] < 2024]
test_data = df[df['year'] == 2024]

# Scale the features
X_train = scaler.fit_transform(train_data[features])
X_test = scaler.transform(test_data[features])

y_train = train_data['tp'].values
y_test = test_data['tp'].values

# Fit linear regression model
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
print(f"\nR² score: {r2:.4f}")
print(f"Number of training samples: {len(train_data)}")
print(f"Number of test samples: {len(test_data)}")
print(f"\nModel Intercept: {model.intercept_:.4f}") 