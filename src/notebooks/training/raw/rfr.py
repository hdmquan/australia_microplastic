#%% Imports
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.datasets import load_and_split_dataset
from src.utils.training import prepare_data, calculate_metrics, plot_predictions
from sklearn.ensemble import RandomForestRegressor

#%% Load and prepare data
train_df, test_df = load_and_split_dataset()

# Prepare features (MODIS bands) and target
feature_cols = [f'modis_sur_refl_b0{i}_1' for i in range(1, 8)]
X_train, X_test, y_train, y_test, y_scaler = prepare_data(
    train_df, test_df, feature_cols, 'TotalPlastics', scale_target=True
)

#%% Create and fit model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=37))
])

model.fit(X_train, y_train)

# Make predictions (on scaled data)
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# Calculate metrics on scaled data first
train_metrics = calculate_metrics(y_train, y_train_pred_scaled)
test_metrics = calculate_metrics(y_test, y_test_pred_scaled)

# Inverse transform both predictions and true values
train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
y_train_original = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Print metrics (these will be in the scaled space)
print(f"Train R² (scaled): {train_metrics['r2']:.3f}, RMSE (scaled): {train_metrics['rmse']:.3f}")
print(f"Test R² (scaled): {test_metrics['r2']:.3f}, RMSE (scaled): {test_metrics['rmse']:.3f}")

#%% Plot actual vs predicted (using original scale data)
fig = plot_predictions(y_train_original, train_pred, y_test_original, test_pred, log_scale=True)
fig.show()


#%%


