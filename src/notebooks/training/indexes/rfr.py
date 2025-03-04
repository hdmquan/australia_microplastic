#%% Imports
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import RandomOverSampler
from src.datasets import load_and_split_dataset

from src.utils.feature_eng import calculate_spectral_indices, add_polynomial_features
from src.utils.training import prepare_data, calculate_metrics, plot_predictions
from src.utils import PATH

#%% Spectral indices calculation
def create_rf_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=1000,
            random_state=37,
            max_depth=15,
            min_samples_leaf=2,
            max_samples=0.7,
            min_samples_split=5,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True
        ))
    ])

#%% Load and prepare data
train_df, test_df = load_and_split_dataset()
train_df = calculate_spectral_indices(train_df)
test_df = calculate_spectral_indices(test_df)

feature_cols = [f'modis_sur_refl_b0{i}_1' for i in range(1, 8)] + \
               ['NDVI', 'NDWI', 'FDI', 'EVI', 'SABI', 'NIRPI', 'NDMI', 'WRI']

X_train, X_test, y_train, y_test, y_scaler = prepare_data(
    train_df, test_df, feature_cols, 'TotalPlastics', scale_target=True
)

#%% Apply Random Oversampling
n_bins = 10
y_binned = pd.qcut(y_train, q=n_bins, labels=False, duplicates='drop')
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_binned_resampled = ros.fit_resample(X_train, y_binned)

bin_to_mean = {bin_val: y_train[y_binned == bin_val].mean() for bin_val in np.unique(y_binned)}
y_train_resampled = np.array([bin_to_mean[bin_val] for bin_val in y_train_binned_resampled])

X_train, y_train = X_train_resampled, y_train_resampled
weights = 1 / np.abs(y_train - np.mean(y_train))
weights = weights / np.sum(weights) * len(weights)

#%% Train initial model and get feature importance
model = create_rf_model()
model.fit(X_train, y_train, regressor__sample_weight=weights)

# Get feature importance for polynomial features
feature_importance = model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=True)

# Save feature importance information
feature_importance_info = {
    'importance_df': importance_df,
    'original_features': feature_cols.copy(),
    'top_n': 3  # Save the number of top features used for polynomial features
}
joblib.dump(feature_importance_info, PATH.WEIGHTS / 'feature_importance_info.joblib')

#%% Enhanced model with polynomial features
X_train_enhanced = add_polynomial_features(X_train, importance_df, update_features=True)
X_test_enhanced = add_polynomial_features(X_test, importance_df)

model = create_rf_model()
model.fit(X_train_enhanced, y_train, regressor__sample_weight=weights)

#%% Make predictions and evaluate
y_train_pred = model.predict(X_train_enhanced)
y_test_pred = model.predict(X_test_enhanced)

train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

print(f"Train R² (scaled): {train_metrics['r2']:.3f}, RMSE (scaled): {train_metrics['rmse']:.3f}")
print(f"Test R² (scaled): {test_metrics['r2']:.3f}, RMSE (scaled): {test_metrics['rmse']:.3f}")

#%% Visualizations
# Predictions plot
train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
y_train_original = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

plot_predictions(y_train_original, train_pred, y_test_original, test_pred, log_scale=True).show()

# Feature importance plot
feature_importance = model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=True)

px.bar(importance_df, 
       x='Importance', 
       y='Feature',
       orientation='h',
       title='Feature Importance').show()

#%% Save model and related information
model_info = {
    'model': model,
    'feature_cols': feature_cols,  # Contains both original and polynomial feature names
    'y_scaler': y_scaler
}
joblib.dump(model_info, PATH.WEIGHTS / 'rf_feat_eng.joblib')

#%% Location-based visualization
# Get original training data coordinates and values
train_locations = px.scatter_mapbox(
    train_df,
    lat='Latitude',
    lon='Longitude',
    size='TotalPlastics',
    color_discrete_sequence=['black'],
    opacity=0.5,
    zoom=2
)
train_locations.update_traces(name='Training Data')

# Calculate prediction error for test data
test_true = y_test_original
test_error = np.abs(test_pred - test_true)
error_normalized = (test_error - test_error.min()) / (test_error.max() - test_error.min())

# Create test predictions visualization
test_locations = go.Figure(go.Scattermapbox(
    lat=test_df['Latitude'],
    lon=test_df['Longitude'],
    mode='markers',
    marker=dict(
        size=test_pred,
        color=error_normalized,
        colorscale=[[0, 'blue'], [1, 'red']],
        showscale=True,
        opacity=0.7
    ),
    text=[f'Predicted: {pred:.2f}<br>Actual: {true:.2f}' 
          for pred, true in zip(test_pred, test_true)],
    hoverinfo='text',
    name='Test Predictions'
))

# Combine plots
combined_plot = go.Figure(data=train_locations.data + test_locations.data)

# Calculate the center and bounds of all data points
all_lats = pd.concat([train_df['Latitude'], test_df['Latitude']])
all_lons = pd.concat([train_df['Longitude'], test_df['Longitude']])
center_lat = (all_lats.max() + all_lats.min()) / 2
center_lon = (all_lons.max() + all_lons.min()) / 2

combined_plot.update_layout(
    mapbox_style="carto-positron",
    title="Training Data and Test Predictions<br>Size: Microplastic concentration, Color: Prediction Error (Test only) Lighter = lower error",
    showlegend=True,
    mapbox=dict(
        center=dict(lat=center_lat, lon=center_lon),
        zoom=2
    ),
    width=800,  # Set width
    height=800  # Set equal height for 1:1 aspect ratio
)

combined_plot.show()

#%%
