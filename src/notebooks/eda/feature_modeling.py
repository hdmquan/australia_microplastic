#%% Imports
import pandas as pd
import holoviews as hv
import numpy as np
from src.utils import PATH
from scipy import stats

# Initialize HoloViews with Matplotlib backend
hv.extension('bokeh')

#%% Define paths and load data
paths = {
    "nettows_info": PATH.RAW_DATA / r"nettows_info.csv",
    "aodn_imos": PATH.RAW_DATA / r"AODN-IMOS Microdebris Data (ALL) from Jan-2021 to Dec-2024.csv",
}

# Load datasets
nettows = pd.read_csv(paths["nettows_info"], 
                      comment=None,
                      header=0)     
aodn = pd.read_csv(paths["aodn_imos"], encoding='latin-1')

#%% Data preparation
# Convert dates to datetime
nettows['DateUTC'] = pd.to_datetime(nettows['DateUTC'], format='%d.%m.%y')
aodn['SAMPLE_DATE'] = pd.to_datetime(aodn['SAMPLE_DATE'], format='%d/%m/%Y')

#%% Missing values and anomalies analysis
print("Nettows Missing Values:")
missing_nettows = nettows.isnull().sum()
print(missing_nettows)
print("\nAODN Missing Values:")
missing_aodn = aodn.isnull().sum()
print(missing_aodn)

# Check for zeros and summarize results
zero_readings = (nettows['TotalPlastics'] == 0).sum()
print(f"\nZero Readings in TotalPlastics: {zero_readings} ({(zero_readings/len(nettows)*100):.2f}%)")

# Identify potential outliers using IQR method
Q1 = nettows['TotalPlastics'].quantile(0.25)
Q3 = nettows['TotalPlastics'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
outliers = nettows[nettows['TotalPlastics'] > outlier_threshold]
print(f"\nPotential outliers in TotalPlastics: {len(outliers)} ({(len(outliers)/len(nettows)*100):.2f}%)")
print(f"Outlier threshold: > {outlier_threshold:.2f}")

nettows.fillna(nettows.select_dtypes(include=['object']).fillna(""), inplace=True)
nettows.fillna(nettows.select_dtypes(exclude=['object']).mean(), inplace=True)
aodn.fillna(aodn.select_dtypes(include=['object']).fillna(""), inplace=True)
aodn.fillna(aodn.select_dtypes(exclude=['object']).mean(), inplace=True)


#%% Zero readings analysis
print("Nettows Zero Readings in TotalPlastics:")
print((nettows['TotalPlastics'] == 0).sum())

#%% Distribution visualization and analysis
# Calculate skewness
skewness = nettows['TotalPlastics'].skew()
print(f"Skewness of TotalPlastics distribution: {skewness:.4f}")

# Test for normality
stat, p_value = stats.shapiro(nettows['TotalPlastics'])
print(f"Shapiro-Wilk test for normality: p-value = {p_value:.6f}")
print(f"The distribution is {'likely normal' if p_value > 0.05 else 'not normal'}")

# Check for multimodality using kernel density estimation
kde = hv.Distribution(nettows['TotalPlastics']).opts(
    width=400, height=300,
    xlabel='Total Plastics', ylabel='Density',
    title='Density Distribution of Total Plastics'
)

# Combine histogram and density plot
dist_nettows = hv.Histogram(np.histogram(nettows['TotalPlastics'], bins=30), label='Distribution of Total Plastics').opts(
    width=400, height=300,
    xlabel='Total Plastics', ylabel='Count',
    title='Distribution of Total Plastics'
)

combined_dist = dist_nettows * kde
combined_dist

#%% Collection method comparison with statistical testing
if 'CollectionMethod' in nettows.columns:
    # Group data by collection method
    method_groups = nettows.groupby('CollectionMethod')['TotalPlastics']
    
    # Generate summary statistics
    method_stats = nettows.groupby('CollectionMethod')['TotalPlastics'].agg(['count', 'mean', 'std', 'median'])
    print("Summary by Collection Method:")
    print(method_stats)
    
    # Statistical testing between methods (if there are at least 2 methods)
    methods = list(method_groups.groups.keys())
    if len(methods) >= 2:
        # For two main methods (e.g., manta vs neuston)
        if len(methods) == 2:
            stat, pval = stats.mannwhitneyu(
                method_groups.get_group(methods[0]), 
                method_groups.get_group(methods[1])
            )
            print(f"\nMann-Whitney U test between {methods[0]} and {methods[1]}: p-value = {pval:.6f}")
            print(f"The difference is {'statistically significant' if pval < 0.05 else 'not statistically significant'}")
        # If more than two methods, use Kruskal-Wallis
        else:
            samples = [group for name, group in method_groups]
            stat, pval = stats.kruskal(*samples)
            print(f"\nKruskal-Wallis test across all methods: p-value = {pval:.6f}")
            print(f"The differences are {'statistically significant' if pval < 0.05 else 'not statistically significant'}")
    
    # Visualization
    method_comparison = hv.BoxWhisker(
        [(method, group['TotalPlastics']) for method, group in nettows.groupby('CollectionMethod')], 
        label='Plastics by Collection Method').opts(
        width=500, height=300,
        xlabel='Collection Method', ylabel='Total Plastics',
        title='Distribution by Collection Method',
        xrotation=45
    )
    
    method_comparison
else:
    print("No 'CollectionMethod' column found in the dataset")

#%% Temporal analysis
# Monthly averages
monthly_avg = nettows.groupby(nettows['DateUTC'].dt.to_period('M'))['TotalPlastics'].mean()

temporal_trend = hv.Curve(
    monthly_avg.reset_index(),
    kdims=['DateUTC'], vdims=['TotalPlastics'],
    label='Monthly Average Plastics'
).opts(
    width=600, height=300,
    xlabel='Date', ylabel='Average Total Plastics',
    title='Temporal Trends in Microplastic Readings'
)

temporal_trend

#%% Summary statistics
print("\nSummary Statistics for Nettows TotalPlastics:")
print(nettows['TotalPlastics'].describe())

#%% Skewness analysis
print("\nSkewness of TotalPlastics distribution:")
print(nettows['TotalPlastics'].skew())

#%% Batch effects analysis
# Create year and month columns for easier analysis
nettows['Year'] = nettows['DateUTC'].dt.year
nettows['Month'] = nettows['DateUTC'].dt.month

# Adjust seasons for Southern Hemisphere
nettows['Season'] = pd.cut(
    nettows['Month'], 
    bins=[0, 2, 5, 8, 11, 12], 
    labels=['Summer', 'Autumn', 'Winter', 'Spring', 'Summer'],
    ordered=False,
    include_lowest=True
)

# Check for team-based differences if team information exists
if 'Team' in nettows.columns:
    team_stats = nettows.groupby('Team')['TotalPlastics'].agg(['count', 'mean', 'std', 'median'])
    print("Summary by Team:")
    print(team_stats)
    
    # Statistical test for team differences
    team_groups = nettows.groupby('Team')['TotalPlastics']
    teams = list(team_groups.groups.keys())
    if len(teams) >= 2:
        samples = [group for name, group in team_groups]
        stat, pval = stats.kruskal(*samples)
        print(f"\nKruskal-Wallis test across teams: p-value = {pval:.6f}")
        print(f"Team differences are {'statistically significant' if pval < 0.05 else 'not statistically significant'}")
    
    # Visualization of team differences
    team_viz = hv.BoxWhisker(
        [(team, group['TotalPlastics']) for team, group in nettows.groupby('Team')], 
        label='Plastics by Team').opts(
        width=500, height=300,
        xlabel='Team', ylabel='Total Plastics',
        title='Distribution by Team',
        xrotation=45
    )
    team_viz

# Season/year analysis for temporal batch effects
season_year_stats = nettows.groupby(['Year', 'Season'])['TotalPlastics'].agg(['count', 'mean', 'std', 'median'])
print("\nSummary by Year and Season:")
print(season_year_stats)


# %%
