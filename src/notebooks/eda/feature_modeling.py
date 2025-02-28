#%% Imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.utils import PATH

df = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")

#%% Zero Analysis
# Count zeros and their distribution
zero_counts = (df['TotalPlastics'] == 0).sum()
total_samples = len(df)
zero_percentage = (zero_counts / total_samples) * 100

# Temporal pattern of zeros
df['Month'] = df['DateUTC'].dt.month
monthly_zeros = df[df['TotalPlastics'] == 0].groupby('Month').size()

# Create subplot figure
fig = make_subplots(rows=1, cols=2, subplot_titles=['Distribution of Measurements by Month', 'Number of Zero Readings by Month'])

# Box plot
fig.add_trace(
    go.Box(x=df['Month'], y=df['TotalPlastics'], name='TotalPlastics'),
    row=1, col=1
)

# Bar plot
fig.add_trace(
    go.Bar(x=monthly_zeros.index, y=monthly_zeros.values, name='Zero Counts'),
    row=1, col=2
)

fig.update_layout(height=500, width=1200, title_text="Zero Analysis", showlegend=False)
fig.show()

print(f"Zero readings: {zero_counts} ({zero_percentage:.2f}% of total samples)")

# Check if zeros are randomly distributed
non_zero_mean = df[df['TotalPlastics'] > 0]['TotalPlastics'].mean()
print(f"\nMean of non-zero readings: {non_zero_mean:.2f}")

#%% Feature Engineering - Time Based
# Time-based features
df['Month'] = df['DateUTC'].dt.month
df['Season'] = pd.cut(df['Month'], 
                     bins=[0, 3, 6, 9, 12], 
                     labels=['Winter', 'Spring', 'Summer', 'Fall'])

#%% Feature Engineering - Spatial
# Basic spatial features
df['DistanceFromEquator'] = abs(df['Latitude'])

# Binned location features
df['LatitudeBin'] = pd.qcut(df['Latitude'], q=5, labels=['Very South', 'South', 'Mid', 'North', 'Very North'])
df['LongitudeBin'] = pd.qcut(df['Longitude'], q=5, labels=['Far West', 'West', 'Mid', 'East', 'Far East'])

#%% Visualization of New Features
# Create subplot figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Plastics by Season',
        'Plastics vs Distance from Equator',
        'Plastics by Latitude Region',
        'Plastics by Longitude Region'
    ]
)

# Season boxplot
fig.add_trace(
    go.Box(x=df['Season'], y=df['TotalPlastics'], name='Season'),
    row=1, col=1
)

# Distance from Equator scatter
fig.add_trace(
    go.Scatter(x=df['DistanceFromEquator'], y=df['TotalPlastics'], 
               mode='markers', name='Distance', marker=dict(size=5)),
    row=1, col=2
)

# Latitude bin boxplot
fig.add_trace(
    go.Box(x=df['LatitudeBin'], y=df['TotalPlastics'], name='Latitude'),
    row=2, col=1
)

# Longitude bin boxplot
fig.add_trace(
    go.Box(x=df['LongitudeBin'], y=df['TotalPlastics'], name='Longitude'),
    row=2, col=2
)

fig.update_layout(height=800, width=1200, showlegend=False)
fig.show()

#%% Statistical Summary of Features
feature_summary = df.groupby('Season')['TotalPlastics'].agg(['mean', 'median', 'std', 'count'])
print("\nSeasonal Statistics:")
print(feature_summary)

spatial_summary = df.groupby('LatitudeBin')['TotalPlastics'].agg(['mean', 'median', 'std', 'count'])
print("\nLatitudinal Statistics:")
print(spatial_summary)
