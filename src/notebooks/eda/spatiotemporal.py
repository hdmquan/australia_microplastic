#%% Imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.utils import PATH

#%% Load and prepare data
df = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")

# Add region classification based on coordinates
def classify_region(row):
    if row['Longitude'] > 142 and row['Latitude'] > -19 and row['Latitude'] < -10:
        return 'GBR'
    elif row['Latitude'] > -39 and row['Latitude'] < -37:
        return 'VIC'
    elif row['Latitude'] < -39:
        return 'TAS'
    else:
        return 'East Ocean'

df['Region'] = df.apply(classify_region, axis=1)

# Add distance from coast (simplified proxy for human activity)
# Consider points closer to coast as < 100km from mainland
def calculate_coastal_distance(row):
    # Simplified distance calculation - can be improved with actual coastline data
    if row['Region'] == 'GBR':
        return 'Coastal'
    elif abs(row['Longitude'] - 150) < 1:  # Rough estimate for east coast
        return 'Coastal'
    else:
        return 'Open Ocean'

df['Coastal_Type'] = df.apply(calculate_coastal_distance, axis=1)

#%% Regional Distribution of Microplastics
fig_region = px.box(
    df,
    x='Region',
    y='TotalPlastics',
    color='Region',
    title='Microplastic Distribution by Region',
    points='all'
)
fig_region.update_layout(
    xaxis_title='Region',
    yaxis_title='Total Plastics Count'
)
fig_region.show()

#%% Temporal Trends
# Add month for seasonality analysis
df['Month'] = df['DateUTC'].dt.month
df['Year'] = df['DateUTC'].dt.year

# Overall temporal trend
fig_temporal = px.scatter(
    df,
    x='DateUTC',
    y='TotalPlastics',
    color='Region',
    trendline="lowess",
    title='Temporal Trends in Microplastic Counts'
)
fig_temporal.update_layout(
    xaxis_title='Date',
    yaxis_title='Total Plastics Count'
)
fig_temporal.show()

#%% Seasonal Patterns
fig_seasonal = px.box(
    df,
    x='Month',
    y='TotalPlastics',
    color='Region',
    title='Seasonal Patterns in Microplastic Counts'
)
fig_seasonal.update_layout(
    xaxis_title='Month',
    yaxis_title='Total Plastics Count'
)
fig_seasonal.show()

#%% Coastal vs Open Ocean Comparison
fig_coastal = px.violin(
    df,
    x='Coastal_Type',
    y='TotalPlastics',
    color='Region',
    box=True,
    title='Microplastic Distribution: Coastal vs Open Ocean'
)
fig_coastal.update_layout(
    xaxis_title='Location Type',
    yaxis_title='Total Plastics Count'
)
fig_coastal.show()

#%% Spatial-Temporal Heat Map
# Get global min and max for consistent scale
z_min = df['TotalPlastics'].min()
z_max = df['TotalPlastics'].max()

fig_heat = px.density_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    z='TotalPlastics',
    animation_frame=df['DateUTC'].dt.to_period('Q').astype(str),
    title='Quarterly Spatial Distribution of Microplastics',
    mapbox_style='carto-positron',
    radius=30,
    range_color=[z_min, z_max]  # Set consistent color scale range
)
fig_heat.update_layout(
    margin={"r":0,"t":30,"l":0,"b":0}
)
fig_heat.show()

#%% Summary Statistics by Region
regional_stats = df.groupby('Region').agg({
    'TotalPlastics': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
print("\nRegional Statistics:")
print(regional_stats)

#%% Year-over-Year Changes
yearly_stats = df.groupby(['Region', 'Year'])['TotalPlastics'].mean().unstack()
print("\nYearly Average by Region:")
print(yearly_stats)
