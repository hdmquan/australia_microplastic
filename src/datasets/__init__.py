import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from pathlib import Path
from src.utils import PATH

def load_and_split_dataset(
    data_path: Path = PATH.PROCESSED_DATA / "nettows_with_modis.parquet",
    test_size: float = 0.2,
    random_state: int = 37
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the dataset while maintaining temporal, spatial, and plastic concentration distribution.
    Uses very coarse binning to ensure sufficient samples in each stratification group.
    
    Args:
        data_path: Path to the parquet file
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Load the dataset
    df = pd.read_parquet(data_path)
    
    # Handle outliers for plastic-related columns
    plastic_columns = ['HardPlastics', 'SoftPlastics', 'PlasticLines', 
                      'Styrofoam', 'Pellets', 'TotalPlastics']
    
    for col in plastic_columns:
        mean = df[col].mean()
        std = df[col].std()
        max_valid = mean + 2 * std
        df[col] = df[col].apply(lambda x: min(x, max_valid))
    
    # Create temporal bins (by year instead of quarter)
    df['year'] = pd.to_datetime(df['DateUTC']).dt.year
    
    # Create very coarse spatial bins (2x2 grid)
    df['lat_bin'] = pd.qcut(df['Latitude'], q=2, labels=False)
    df['lon_bin'] = pd.qcut(df['Longitude'], q=2, labels=False)
    
    # Even simpler plastic concentration bins
    def get_plastic_bin(x):
        if x == 0:
            return 0  # Zero counts
        else:
            return 1  # Non-zero counts
    
    df['plastic_bin'] = df['TotalPlastics'].apply(get_plastic_bin)
    
    # Combine all stratification factors
    df['strat_group'] = (df['year'].astype(str) + '_' + 
                        df['lat_bin'].astype(str) + '_' + 
                        df['lon_bin'].astype(str) + '_' + 
                        df['plastic_bin'].astype(str))
    
    # Count samples in each group and remove rare groups
    group_counts = df['strat_group'].value_counts()
    valid_groups = group_counts[group_counts >= 5].index
    df = df[df['strat_group'].isin(valid_groups)]
    
    # Check if we have enough samples for the split
    n_groups = len(df['strat_group'].unique())
    min_test_size = max(0.2, n_groups / len(df))
    test_size = max(test_size, min_test_size)
    
    # Perform stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['strat_group']
    )
    
    # Clean up temporary columns
    for df_ in [train_df, test_df]:
        df_.drop(['year', 'lat_bin', 'lon_bin', 
                 'plastic_bin', 'strat_group'], axis=1, inplace=True)
    
    print(f"Train set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df
