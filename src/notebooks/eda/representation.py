#%% Imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.utils import PATH

#%% Load and prepare data
df = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")

# I need to... manually classify the regions and it's wrong :((
def classify_region(row):
    if row['Longitude'] > 142 and row['Latitude'] > -18:  # GBR
        return 'GBR'
    elif row['Latitude'] > -40 and row['Latitude'] < -35 and row['Longitude'] > 145:  # VIC
        return 'VIC'
    elif row['Latitude'] < -40:  # TAS
        return 'TAS'
    elif row['Longitude'] < 130:  # Western Australia
        return 'WA'
    else:
        return 'East Ocean'

df['Region'] = df.apply(classify_region, axis=1)

#%% Temporal Sampling Distribution
fig_temporal = px.histogram(
    df,
    x='DateUTC',
    color='Region',
    title='Temporal Distribution of Sampling',
    nbins=50
)
fig_temporal.update_layout(
    xaxis_title='Date',
    yaxis_title='Number of Samples'
)
fig_temporal.show()

#%% Spatial Coverage Analysis
fig_spatial = px.scatter_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    color='Region',
    size='TotalPlastics',
    hover_data=['DateUTC', 'TotalPlastics'],
    title='Spatial Distribution of Sampling Locations',
    mapbox_style='carto-positron'
)
fig_spatial.update_layout(
    margin={"r":0,"t":30,"l":0,"b":0}
)
fig_spatial.show()

#%% Plastic Type Distribution
plastic_types = ['HardPlastics', 'SoftPlastics', 'PlasticLines', 'Styrofoam', 'Pellets']
plastic_long = df[plastic_types].melt()

fig_types = px.box(
    plastic_long,
    x='variable',
    y='value',
    title='Distribution of Different Plastic Types',
    points='all'
)
fig_types.update_layout(
    xaxis_title='Plastic Type',
    yaxis_title='Count'
)
fig_types.show()

#%% Extreme Events Analysis
# Calculate Z-scores for TotalPlastics
df['Plastic_Zscore'] = (df['TotalPlastics'] - df['TotalPlastics'].mean()) / df['TotalPlastics'].std()

# Plot distribution with extreme events highlighted
fig_extreme = px.histogram(
    df,
    x='TotalPlastics',
    color=df['Plastic_Zscore'].abs() > 2,  # Highlight samples > 2 std dev
    title='Distribution of Total Plastic Counts (Extreme Events Highlighted)',
    nbins=50,
    color_discrete_map={True: 'red', False: 'blue'},
    labels={'color': 'Extreme Event (>2σ)'}
)
fig_extreme.show()

#%% Sampling Frequency by Region and Time
# Create a pivot table of sampling frequency using datetime strings
df['YearMonth'] = df['DateUTC'].dt.strftime('%Y-%m')
temporal_coverage = pd.crosstab(
    df['YearMonth'],
    df['Region'],
    margins=True
)

# Plot heatmap of sampling frequency
fig_coverage = px.imshow(
    temporal_coverage.iloc[:-1, :-1],  # Exclude margins
    title='Sampling Frequency Heatmap by Region and Time',
    labels=dict(x='Region', y='Time', color='Number of Samples')
)
fig_coverage.show()

#%% Summary Statistics
print("\nSampling Statistics by Region:")
region_stats = df.groupby('Region').agg({
    'TotalPlastics': ['count', 'mean', 'std'],
}).round(2)
print(region_stats)

print("\nExtreme Events Summary:")
extreme_events = df[df['Plastic_Zscore'].abs() > 2]
print(f"Number of extreme events (>2σ): {len(extreme_events)}")
print(f"Percentage of total samples: {(len(extreme_events)/len(df)*100):.2f}%")

print("\nPlastic Type Composition:")
plastic_composition = df[plastic_types].agg(['mean', 'std', 'min', 'max']).round(2)
print(plastic_composition)
