#%% Imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from src.utils import PATH

#%% Load processed data
df = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")

#%% Missing Values Analysis
missing_data = df.isnull().sum()
print("Missing Values Summary:")
print(missing_data)

zero_counts = (df == 0).sum()
print("\nZero Value Counts:")
print(zero_counts)

#%% Distribution of Total Plastics
fig_hist = px.histogram(
    df, 
    x='TotalPlastics',
    nbins=30,
    title='Distribution of Total Plastic Counts'
)
fig_hist.update_layout(
    xaxis_title='Total Plastics Count',
    yaxis_title='Frequency'
)
fig_hist.show()

#%% Distribution of Different Plastic Types (Box Plot)
plastic_types = ['HardPlastics', 'SoftPlastics', 'PlasticLines', 'Styrofoam', 'Pellets']
fig_box = px.box(
    df,
    y=plastic_types,
    title='Distribution of Different Plastic Types'
)
fig_box.update_layout(
    yaxis_title='Plastic Type',
    xaxis_title='Count'
)
fig_box.show()

#%% Spatial Distribution of Total Plastics
# Convert dates to numeric values (days since the earliest date)
df['days_since_start'] = (df['DateUTC'] - df['DateUTC'].min()).dt.total_seconds() / (24 * 60 * 60)

fig_scatter = px.scatter_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    color='days_since_start',
    size='TotalPlastics',
    hover_data=['DateUTC', 'TotalPlastics'],
    title='Spatial Distribution of Total Plastics',
    mapbox_style='carto-positron',
    color_continuous_scale='cividis_r'
)
fig_scatter.update_layout(
    margin={"r":0,"t":30,"l":0,"b":0},
    coloraxis_colorbar_title="Days Since First Sample"
)
fig_scatter.show()

#%% Time Series Analysis
fig_time = px.scatter(
    df,
    x='DateUTC',
    y='TotalPlastics',
    title='Total Plastics Over Time'
)
fig_time.update_layout(
    xaxis_title='Date',
    yaxis_title='Total Plastics'
)
fig_time.show()

#%% Correlation Matrix
correlation = df.corr()

fig_corr = px.imshow(
    correlation,
    labels=dict(color="Correlation"),
    title='Correlation'
)
fig_corr.update_layout(
    width=600,
    height=600
)
fig_corr.show()

#%% Distribution by Plastic Type (Violin Plot)
fig_violin = px.violin(
    df.melt(value_vars=plastic_types),
    y='value',
    x='variable',
    box=True,
    title='Distribution by Plastic Type (Violin Plot)'
)
fig_violin.update_layout(
    xaxis_title='Plastic Type',
    yaxis_title='Count'
)
fig_violin.show()
