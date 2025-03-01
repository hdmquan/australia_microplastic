import pandas as pd
import numpy as np
from src.utils import PATH

def process_nettows(df):
    """Process nettows dataset.
    Keep only location data and plastic counts."""
    # Convert dates to datetime
    df['DateUTC'] = pd.to_datetime(df['DateUTC'], format='%d.%m.%y')
    
    # Average start and end coordinates
    df['Longitude'] = (df['StartLongitude'] + df['EndLongitude']) / 2
    df['Latitude'] = (df['StartLatitude'] + df['EndLatitude']) / 2
    
    # Keep only relevant columns
    cols_to_keep = [
        'DateUTC', 'Longitude', 'Latitude',
        'HardPlastics', 'SoftPlastics', 'PlasticLines', 
        'Styrofoam', 'Pellets', 'TotalPlastics'
    ]
    
    return df[cols_to_keep]

def process_aodn(df):
    """Process AODN-IMOS dataset."""
    # Convert dates to datetime
    df['SAMPLE_DATE'] = pd.to_datetime(df['SAMPLE_DATE'], format='%d/%m/%Y')
    
    # TODO: Add specific processing for AODN dataset
    # This will need to be expanded based on the AODN data structure
    return df

def main():
    # Define paths
    paths = {
        "nettows_info": PATH.RAW_DATA / "nettows_info.csv",
        "aodn_imos": PATH.RAW_DATA / "AODN-IMOS Microdebris Data (ALL) from Jan-2021 to Dec-2024.csv",
    }
    
    # Load and process nettows data
    nettows = pd.read_csv(paths["nettows_info"], comment=None, header=0)
    nettows_processed = process_nettows(nettows)
    
    # Load and process AODN data
    aodn = pd.read_csv(paths["aodn_imos"], encoding='latin-1')
    aodn_processed = process_aodn(aodn)
    
    # TODO: Combine datasets
    # This will need to be implemented once we know the structure of both processed datasets
    
    # Save processed data
    output_path = PATH.PROCESSED_DATA / "nettows_processed.parquet"
    nettows_processed.to_parquet(output_path)

if __name__ == "__main__":
    main()
