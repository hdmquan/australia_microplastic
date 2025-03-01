#%% Imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import PATH

#%% Load the dataset
# Load the processed dataset from parquet file
data_path = PATH.PROCESSED_DATA / "nettows_with_modis.parquet"
df = pd.read_parquet(data_path)

# Extract the columns into numpy arrays for consistency with rest of code
modis_b1 = df['modis_sur_refl_b01_1'].values
modis_b2 = df['modis_sur_refl_b02_1'].values
modis_b3 = df['modis_sur_refl_b03_1'].values
modis_b4 = df['modis_sur_refl_b04_1'].values
modis_b5 = df['modis_sur_refl_b05_1'].values
modis_b6 = df['modis_sur_refl_b06_1'].values
modis_b7 = df['modis_sur_refl_b07_1'].values
plastic_amount = df['TotalPlastics'].values

#%% Calculate NDVI and FDI
# NDVI = (NIR - Red) / (NIR + Red)
# For MODIS, band 2 is NIR (841-876 nm) and band 1 is Red (620-670 nm)
ndvi = (modis_b2 - modis_b1) / (modis_b2 + modis_b1)

# FDI = NIR - (Red + SWIR)/2
# For MODIS, band 6 is SWIR (1628-1652 nm)
fdi = modis_b2 - (modis_b1 + modis_b6) / 2

#%% Prepare data for correlation analysis
# Flatten arrays and create DataFrame
df = pd.DataFrame({
    'Band1': modis_b1.flatten(),
    'Band2': modis_b2.flatten(),
    'Band3': modis_b3.flatten(),
    'Band4': modis_b4.flatten(),
    'Band5': modis_b5.flatten(),
    'Band6': modis_b6.flatten(),
    'Band7': modis_b7.flatten(),
    'NDVI': ndvi.flatten(),
    'FDI': fdi.flatten(),
    'Plastic_Amount': plastic_amount.flatten()
})

#%% Calculate correlations
correlations = df.corr()['Plastic_Amount'].sort_values(ascending=False)

#%% Plot correlation heatmap
# Create mask for upper triangle
mask = np.triu(np.ones_like(df.corr()))

fig = go.Figure(data=go.Heatmap(
    x=df.columns,
    y=df.columns,
    colorscale='RdBu',
    zmin=-1,
    zmax=1,
    # Set values in lower triangle to None to hide them
    z=np.where(mask, df.corr(), None)
))

fig.update_layout(
    title='Correlation Matrix of MODIS Bands, Indices, and Plastic Amount',
    width=800,
    height=800
)
fig.show()

#%% Plot bar chart of correlations with plastic amount
fig = go.Figure(data=go.Bar(
    x=correlations.index[1:],
    y=correlations.values[1:],
    text=np.round(correlations.values[1:], 3),
    textposition='auto',
))

fig.update_layout(
    title='Correlation with Plastic Amount',
    xaxis_title='Features',
    yaxis_title='Correlation Coefficient',
    yaxis=dict(range=[-1, 1]),
    width=800,
    height=500
)
fig.show()

#%% Scatter plots for top 3 correlated features
for feature in correlations.index[1:4]:
    fig = px.scatter(df.sample(100),
                    x=feature,
                    y='Plastic_Amount',
                    title=f'{feature} vs Plastic Amount',
                    opacity=0.5)
    fig.show()

#%% Distribution plots
fig = go.Figure()
for col in df.columns:
    fig.add_trace(go.Histogram(
        x=df[col],
        name=col,
        opacity=0.7,
        nbinsx=50
    ))

fig.update_layout(
    title='Distribution of Features',
    barmode='overlay',
    width=800,
    height=500
)
fig.show()

#%% Analyze
for band in range(1, 8):
    modis_b = df[f'Band{band}'].values
    print(f"Band {band} Statistics:")
    print(f"Min: {modis_b.min():.3f}")
    print(f"Max: {modis_b.max():.3f}")
    print(f"Mean: {modis_b.mean():.3f}")
    print(f"Number of negative values: {np.sum(modis_b < 0)}")
    print(f"Percentage of negative values: {(np.sum(modis_b < 0) / len(modis_b) * 100):.2f}%")


# %%
