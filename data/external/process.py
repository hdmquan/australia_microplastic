import pandas as pd
import numpy as np
from pathlib import Path
import pyhdf.SD as sd
from loguru import logger
from datetime import datetime

from src.utils import PATH

def get_modis_file_for_date(date, modis_dir):
    """Find the corresponding MODIS file for a given month."""
    modis_files = list(modis_dir.glob("MYD09GA.A*.hdf"))
    
    if not modis_files:
        logger.error(f"No MODIS files found in directory: {modis_dir}")
        return None
    
    # Get the first day of the month and last day of the month
    month_start = date.replace(day=1)
    next_month = (date.replace(day=1) + pd.DateOffset(months=1))
    month_end = next_month - pd.DateOffset(days=1)
    
    matching_files = []
    for file in modis_files:
        try:
            year_doy = file.name.split('.')[1][1:]  # YYYYDDD
            file_date = datetime.strptime(year_doy, "%Y%j")
            
            if month_start <= file_date <= month_end:
                matching_files.append((file_date, file))
                
        except (ValueError, IndexError):
            continue
    
    if not matching_files:
        logger.warning(f"No matching files found for {date.strftime('%Y-%m')}")
        return None
    
    # Sort by date and take the middle of the month
    matching_files.sort()
    chosen_file = matching_files[len(matching_files)//2][1]
    logger.info(f"Selected file {chosen_file.name} for {date.strftime('%Y-%m')}")
    
    return chosen_file

def extract_modis_bands(hdf_file):
    """Extract specific MODIS bands (b01_1 through b07_1) from HDF file"""
    hdf = sd.SD(str(hdf_file))
    
    try:
        datasets = hdf.datasets()
        bands_data = {}
        
        # Only process bands 1-7 with suffix '_1'
        target_bands = [f'sur_refl_b0{i}_1' for i in range(1, 8)]
        
        for ds_name in datasets:
            try:
                if ds_name in target_bands:
                    dataset = hdf.select(ds_name)
                    data = dataset[:]
                    scale_factor = dataset.attributes().get('scale_factor', 1)
                    
                    # Apply scale factor if available
                    if scale_factor != 1:
                        data = data.astype(float) * scale_factor
                    
                    bands_data[ds_name] = data
            except Exception as e:
                logger.warning(f"Failed to extract band {ds_name}: {str(e)}")
                continue
                
        return bands_data
    
    finally:
        hdf.end()

def get_nearest_pixel_value(lat, lon, band_data):
    """Get pixel value using simple nearest neighbor approach"""
    # Get image dimensions
    if not isinstance(band_data, np.ndarray) or band_data.ndim != 2:
        logger.warning(f"Invalid band data shape: {band_data.shape if hasattr(band_data, 'shape') else 'unknown'}")
        return np.nan
        
    rows, cols = band_data.shape
    
    # Convert lat/lon to pixel coordinates
    # MODIS grid is 2400x2400 pixels for each tile
    # Each tile covers 10 degrees latitude x 10 degrees longitude
    
    # Assuming standard MODIS tile boundaries
    tile_size = 10.0  # degrees
    row_scale = rows / tile_size
    col_scale = cols / tile_size
    
    # Convert to pixel coordinates
    # Note: This is a simplified transformation - may need adjustment based on tile boundaries
    try:
        i = int(np.round((90 - lat) * row_scale)) % rows
        j = int(np.round((lon + 180) * col_scale)) % cols
        
        # Check bounds
        if 0 <= i < rows and 0 <= j < cols:
            value = band_data[i, j]
            # Check for fill values
            if value > 32000:  # MODIS typical fill value
                return np.nan
            return value
        return np.nan
    except Exception as e:
        logger.warning(f"Error calculating pixel coordinates for {lat}, {lon}: {str(e)}")
        return np.nan

def main():
    logger.info("Starting MODIS data processing")
    
    mp_data = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")
    mp_data['_month'] = pd.to_datetime(mp_data['DateUTC']).dt.to_period('M')
    
    modis_values = {}
    modis_dir = PATH.DATA / "external" / "modis"
    
    unique_months = mp_data['_month'].unique()
    logger.info(f"Processing {len(unique_months)} unique months")
    
    for month in unique_months:
        logger.info(f"Processing month: {month}")
        
        modis_file = get_modis_file_for_date(month.to_timestamp(), modis_dir)
        if modis_file is None:
            continue
        
        try:
            bands_data = extract_modis_bands(modis_file)
            if not bands_data:
                logger.warning(f"No valid bands extracted from {modis_file}")
                continue
                
            logger.info(f"Processing {len(bands_data)} bands")
            
            month_mask = mp_data['_month'] == month
            month_data = mp_data[month_mask].copy()  # Add .copy() to prevent SettingWithCopyWarning
            
            for band_name, band_data in bands_data.items():
                col_name = f"modis_{band_name}"
                if col_name not in modis_values:
                    modis_values[col_name] = np.full(len(mp_data), np.nan)
                
                # Process each row individually to avoid the unpacking error
                for idx, row in month_data.iterrows():
                    try:
                        value = get_nearest_pixel_value(row['Latitude'], row['Longitude'], band_data)
                        # Find the position in the original dataframe
                        original_idx = mp_data.index.get_loc(idx)
                        modis_values[col_name][original_idx] = value
                    except Exception as e:
                        logger.warning(f"Error processing pixel at {row['Latitude']}, {row['Longitude']}: {str(e)}")
                        continue
            
        except Exception as e:
            logger.error(f"Failed to process {modis_file}: {str(e)}")
            continue
    
    # Add MODIS columns to dataframe
    logger.info("Adding MODIS bands to dataframe")
    for col_name, values in modis_values.items():
        mp_data[col_name] = values
    
    # Remove temporary month column
    mp_data = mp_data.drop(columns=['_month'])
    
    # Save merged dataset
    output_file = PATH.PROCESSED_DATA / "nettows_with_modis.parquet"
    mp_data.to_parquet(output_file)
    logger.info(f"Saved merged dataset to {output_file}")
    
    # Log coverage statistics
    non_null_counts = mp_data.filter(like='modis_').notna().sum()
    logger.info("MODIS data coverage:")
    for col, count in non_null_counts.items():
        logger.info(f"{col}: {count}/{len(mp_data)} values ({count/len(mp_data)*100:.1f}%)")

if __name__ == "__main__":
    main()
