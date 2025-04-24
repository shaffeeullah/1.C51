import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Read the data
print("Reading data...")
df = pd.read_csv('data/monthly_weather_summary.csv')

# Sort data by year and month
df = df.sort_values(['year', 'month'])

# Calculate correlation matrix for specific features
correlation_features = ['u10', 't2m', 'tcc']
correlation_matrix = df[correlation_features].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Calculate p-values
p_values = pd.DataFrame(np.zeros((len(correlation_features), len(correlation_features))), 
                       index=correlation_features, 
                       columns=correlation_features)

for i in range(len(correlation_features)):
    for j in range(len(correlation_features)):
        if i != j:
            corr, p_value = stats.pearsonr(df[correlation_features[i]], df[correlation_features[j]])
            p_values.iloc[i, j] = p_value
        else:
            p_values.iloc[i, j] = 0  # p-value is 0 for self-correlation

print("\nP-values Matrix:")
print(p_values)

print("\nSignificance levels:")
print("p < 0.001: ***")
print("p < 0.01: **")
print("p < 0.05: *")
print("p >= 0.05: ns (not significant)")

# Create a heatmap visualization with significance stars
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Weather Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Create lag features (previous year's data)
df['prev_year_tp'] = df.groupby('month')['tp'].shift(1)
df['prev_year_t2m'] = df.groupby('month')['t2m'].shift(1)
df['prev_year_tcc'] = df.groupby('month')['tcc'].shift(1)
df['prev_year_u10'] = df.groupby('month')['u10'].shift(1)

# Drop rows with NaN values (first year will have NaN for lag features)
df = df.dropna()

# Define numerical and categorical features
numerical_features = ['latitude', 'longitude', 'u10', 't2m', 'tcc', 
                     'prev_year_tp', 'prev_year_u10', 'prev_year_t2m', 'prev_year_tcc']
categorical_features = ['month']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ])

# Initialize lists to store results
r2_scores = []
insufficient_data = 0
total_groups = 0

print("Performing regression analysis...")
# Split into training and test sets (using last year as test)
train_data = df[df['year'] < df['year'].max()]
test_data = df[df['year'] == df['year'].max()]

# Create feature matrix
features = numerical_features + categorical_features

# Transform the features
X_train = preprocessor.fit_transform(train_data[features])
X_test = preprocessor.transform(test_data[features])

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

# Get feature names after preprocessing
feature_names = (numerical_features + 
                [f"month_{i}" for i in range(1, len(preprocessor.named_transformers_['cat'].categories_[0])+1)])

# Calculate p-values for regression coefficients
n = len(y_train)
p = len(feature_names)
dof = n - p - 1  # degrees of freedom

# Calculate standard errors
mse = np.sum((y_train - model.predict(X_train)) ** 2) / dof
var_b = mse * (np.linalg.inv(np.dot(X_train.T, X_train)).diagonal())
sd_b = np.sqrt(var_b)

# Calculate t-statistics and p-values
t_stat = model.coef_ / sd_b
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))

# Print feature importance with p-values
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_,
    'P-value': p_values
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance with P-values:")
print(feature_importance)
print(f"\nR² score: {r2:.4f}")
print(f"Number of training samples: {len(train_data)}")
print(f"Number of test samples: {len(test_data)}")
print(f"\nModel Intercept: {model.intercept_:.4f}") 