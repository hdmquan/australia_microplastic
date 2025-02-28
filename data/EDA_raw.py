#%% Imports
import pandas as pd
import holoviews as hv
from src.utils import PATH
import numpy as np

# Initialize HoloViews with Matplotlib backend
hv.extension('bokeh')

#%% Define paths and load data
paths = {
    "nettows_info": PATH.RAW_DATA / r"nettows_info.csv",
    "aodn_imos": PATH.RAW_DATA / r"AODN-IMOS Microdebris Data (ALL) from Jan-2021 to Dec-2024.csv",
}

# Load datasets
nettows = pd.read_csv(paths["nettows_info"], 
                      comment=None,  # First character is a hash so pandas doesn't read it as a comment
                      header=0)     

# For some reason, the AODN file is encoded in latin-1?
aodn = pd.read_csv(paths["aodn_imos"], encoding='latin-1')

#%%
nettows.info()

#%%
aodn.info()

#%% Clean and prepare data
# Convert dates to datetime
nettows['DateUTC'] = pd.to_datetime(nettows['DateUTC'], format='%d.%m.%y')
aodn['SAMPLE_DATE'] = pd.to_datetime(aodn['SAMPLE_DATE'], format='%d/%m/%Y')

#%% Visualization 1: Sampling locations map
locations_nettows = hv.Points(
    data=nettows, 
    kdims=['StartLongitude', 'StartLatitude'],
    label='Nettows Sampling Locations'
).opts(color='blue', size=8, title='Sampling Locations')

locations_aodn = hv.Points(
    data=aodn, 
    kdims=['START_LONG', 'START_LAT'],
    label='AODN-IMOS Sampling Locations'
).opts(color='red', size=8)

combined_map = (locations_nettows * locations_aodn).opts(
    width=600, height=400,
    xlabel='Longitude', ylabel='Latitude'
)

combined_map

#%% Visualization 2: Plastic counts over time (Nettows)
plastic_time = hv.Points(
    data=nettows, 
    kdims=['DateUTC', 'TotalPlastics'],
    label='Plastic Counts Over Time'
).opts(
    width=600, height=300,
    xlabel='Date', ylabel='Total Plastics',
    title='Plastic Counts Over Time (Nettows)'
)

plastic_time

#%% Visualization 3: AODN polymer types distribution (Top 10)
polymer_dist = hv.Bars(
    aodn['POLYMER_TYPE'].value_counts().nlargest(10),
    label='Polymer Types Distribution'
).opts(
    width=600, height=400,
    xlabel='Polymer Type', ylabel='Count',
    title='Distribution of Polymer Types (Top 10)',
    xrotation=45
)

polymer_dist

#%%