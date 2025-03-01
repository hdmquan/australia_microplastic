import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Any, Optional

def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    scale_target: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """Prepare features and target for training."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    y_scaler = None
    if scale_target:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    
    return X_train, X_test, y_train, y_test, y_scaler

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }

def plot_predictions(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    log_scale: bool = True
) -> go.Figure:
    """Create actual vs predicted plot."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Training Set', 'Test Set'],
        horizontal_spacing=0.1
    )
    
    
    # Add traces for training data
    fig.add_trace(
        go.Scatter(x=y_train, y=y_train_pred, mode='markers',
                  name='Training Data', marker=dict(opacity=0.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[min(y_train), max(y_train)], y=[min(y_train), max(y_train)],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Add traces for test data
    fig.add_trace(
        go.Scatter(x=y_test, y=y_test_pred, mode='markers',
                  name='Test Data', marker=dict(opacity=0.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1
    )

    fig.update_layout(
        height=800,  # Increased height for better square aspect ratio
        width=500,   # Made width equal to height
        title_text="Model Predictions vs Actual Values",
        showlegend=True
    )
    
    # Add axis labels
    fig.update_xaxes(title_text="Actual Values", row=1, col=1)
    fig.update_xaxes(title_text="Actual Values", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Values", row=2, col=1)
    
    return fig