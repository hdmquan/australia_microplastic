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
                    attrs = dataset.attributes()
                    
                    # Get scale factor and offset
                    scale_factor = attrs.get('scale_factor', 1)
                    add_offset = attrs.get('add_offset', 0)
                    
                    # Get fill value
                    fill_value = attrs.get('_FillValue', 32767)  # Default MODIS fill value
                    
                    # Convert to float and mask fill values
                    data = data.astype(float)
                    data[data == fill_value] = np.nan
                    
                    # Apply scale factor and offset
                    data = data * scale_factor + add_offset
                    
                    # Apply valid range check (reflectance should be 0-1)
                    valid_min = attrs.get('valid_min', 0)
                    valid_max = attrs.get('valid_max', 1)
                    data[data < valid_min] = np.nan
                    data[data > valid_max] = np.nan
                    
                    bands_data[ds_name] = data
                    
                    # Log statistics for debugging
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        logger.info(f"Band {ds_name} stats - Min: {valid_data.min():.4f}, "
                                   f"Max: {valid_data.max():.4f}, Mean: {valid_data.mean():.4f}, "
                                   f"NaN: {np.isnan(data).sum()} pixels")
                
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

def impute_modis_temporal(df):
    """Impute missing MODIS values using temporal interpolation"""
    # Create a copy to avoid warnings
    df_imputed = df.copy()
    
    # Sort by location and time
    df_imputed = df_imputed.sort_values(['Latitude', 'Longitude', 'DateUTC'])
    
    # Get all MODIS band columns
    modis_cols = [col for col in df_imputed.columns if col.startswith('modis_sur_refl_b')]
    
    # Group by location and interpolate within each group
    logger.info("Performing temporal interpolation for MODIS bands")
    
    # Group by location (rounded to 2 decimal places for spatial proximity)
    df_imputed['lat_bin'] = np.round(df_imputed['Latitude'], 2)
    df_imputed['lon_bin'] = np.round(df_imputed['Longitude'], 2)
    
    # Apply interpolation to each spatial group
    for (lat, lon), group in df_imputed.groupby(['lat_bin', 'lon_bin']):
        if len(group) > 1:  # Need at least 2 points for interpolation
            # Get indices in the original dataframe
            idx = group.index
            
            # Interpolate MODIS bands (linear is most appropriate)
            interpolated = group[modis_cols].interpolate(method='linear', limit=3)
            
            # Update the values in the main dataframe
            df_imputed.loc[idx, modis_cols] = interpolated
    
    # Remove temporary columns
    df_imputed = df_imputed.drop(columns=['lat_bin', 'lon_bin'])
    
    # Log imputation statistics
    for col in modis_cols:
        before = df[col].isna().sum()
        after = df_imputed[col].isna().sum()
        filled = before - after
        logger.info(f"Imputed {filled} values in {col} ({filled/len(df)*100:.1f}%)")
    
    return df_imputed

def impute_modis_spatial(df, date_col='DateUTC', time_window='M'):
    """Impute missing MODIS values using spatial interpolation within time windows"""
    from scipy.interpolate import griddata
    
    df_imputed = df.copy()
    modis_cols = [col for col in df.columns if col.startswith('modis_sur_refl_b')]
    
    # Add time period column for grouping
    df_imputed['time_period'] = pd.to_datetime(df_imputed[date_col]).dt.to_period(time_window)
    
    # Process each time period separately
    for period, period_df in df_imputed.groupby('time_period'):
        logger.info(f"Processing spatial interpolation for period {period}")
        
        # For each MODIS band
        for col in modis_cols:
            # Skip if no valid data in this period
            if period_df[col].notna().sum() < 3:  # Need at least 3 points for 2D interpolation
                continue
                
            # Get points with valid data
            valid_mask = period_df[col].notna()
            points = period_df.loc[valid_mask, ['Latitude', 'Longitude']].values
            values = period_df.loc[valid_mask, col].values
            
            # Points to interpolate
            missing_mask = period_df[col].isna()
            if missing_mask.sum() == 0:
                continue
                
            xi = period_df.loc[missing_mask, ['Latitude', 'Longitude']].values
            
            # Perform interpolation (use 'linear' or 'cubic')
            try:
                interpolated = griddata(points, values, xi, method='linear')
                
                # Update values in the dataframe
                period_indices = period_df.index[missing_mask]
                df_imputed.loc[period_indices, col] = interpolated
            except Exception as e:
                logger.warning(f"Spatial interpolation failed for {col} in period {period}: {e}")
    
    # Remove temporary column
    df_imputed = df_imputed.drop(columns=['time_period'])
    
    # Log results
    for col in modis_cols:
        before = df[col].isna().sum()
        after = df_imputed[col].isna().sum()
        filled = before - after
        logger.info(f"Spatially imputed {filled} values in {col} ({filled/len(df)*100:.1f}%)")
    
    return df_imputed

def impute_modis_band_correlation(df):
    """Impute missing values using correlations between bands"""
    from sklearn.ensemble import RandomForestRegressor
    
    df_imputed = df.copy()
    modis_cols = [col for col in df.columns if col.startswith('modis_sur_refl_b')]
    
    # For each band
    for target_col in modis_cols:
        # Skip if band has no missing values
        if df_imputed[target_col].isna().sum() == 0:
            continue
            
        logger.info(f"Imputing {target_col} using other bands")
        
        # Find rows where target is missing but other bands have data
        missing_mask = df_imputed[target_col].isna()
        
        # Create feature matrix from other bands
        feature_cols = [col for col in modis_cols if col != target_col]
        
        # Train on rows where target is not missing
        train_mask = ~missing_mask
        if train_mask.sum() < 10:  # Need enough training data
            logger.warning(f"Not enough training data for {target_col}")
            continue
            
        # Check if we have enough complete rows for training
        X_train = df_imputed.loc[train_mask, feature_cols]
        if X_train.isna().any(axis=1).all():
            logger.warning(f"No complete feature rows for training {target_col}")
            continue
            
        # Drop rows with NaN in features
        valid_train = ~X_train.isna().any(axis=1)
        X_train = X_train[valid_train]
        y_train = df_imputed.loc[train_mask, target_col][valid_train]
        
        if len(X_train) < 10:
            logger.warning(f"Not enough complete training samples for {target_col}")
            continue
            
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict for rows with missing target but complete features
        to_predict = missing_mask & ~df_imputed[feature_cols].isna().any(axis=1)
        if to_predict.sum() > 0:
            X_pred = df_imputed.loc[to_predict, feature_cols]
            predictions = model.predict(X_pred)
            
            # Ensure predictions are within valid range
            predictions = np.clip(predictions, 0, 1)
            
            # Update values
            df_imputed.loc[to_predict, target_col] = predictions
            
            logger.info(f"Imputed {to_predict.sum()} values in {target_col}")
    
    return df_imputed

def impute_modis_combined(df):
    """Apply multiple imputation methods in sequence"""
    # First try temporal interpolation (most accurate)
    df_imputed = impute_modis_temporal(df)
    
    # Then try spatial interpolation for remaining missing values
    df_imputed = impute_modis_spatial(df_imputed)
    
    # Finally use band correlation for any remaining gaps
    df_imputed = impute_modis_band_correlation(df_imputed)
    
    # For any remaining missing values, use monthly means
    modis_cols = [col for col in df_imputed.columns if col.startswith('modis_sur_refl_b')]
    df_imputed['month'] = pd.to_datetime(df_imputed['DateUTC']).dt.month
    
    for col in modis_cols:
        # Calculate monthly means for each band
        monthly_means = df_imputed.groupby('month')[col].mean()
        
        # Fill remaining NaNs with monthly means
        for month, mean_val in monthly_means.items():
            month_mask = (df_imputed['month'] == month) & (df_imputed[col].isna())
            df_imputed.loc[month_mask, col] = mean_val
    
    # Remove temporary column
    df_imputed = df_imputed.drop(columns=['month'])
    
    # Log final imputation statistics
    for col in modis_cols:
        remaining = df_imputed[col].isna().sum()
        if remaining > 0:
            logger.warning(f"{col} still has {remaining} missing values ({remaining/len(df_imputed)*100:.1f}%)")
        else:
            logger.info(f"{col} successfully imputed all values")
    
    # Final fallback - use global mean for any remaining missing values
    for col in modis_cols:
        if df_imputed[col].isna().sum() > 0:
            global_mean = df_imputed[col].mean()
            if not np.isnan(global_mean):
                df_imputed[col].fillna(global_mean, inplace=True)
                logger.info(f"Filled remaining {col} values with global mean: {global_mean:.4f}")
    
    return df_imputed

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
    
    # Log coverage statistics before imputation
    non_null_counts = mp_data.filter(like='modis_').notna().sum()
    logger.info("MODIS data coverage before imputation:")
    for col, count in non_null_counts.items():
        logger.info(f"{col}: {count}/{len(mp_data)} values ({count/len(mp_data)*100:.1f}%)")
    
    # Apply imputation
    logger.info("Applying imputation to fill missing MODIS values")
    mp_data = impute_modis_combined(mp_data)
    
    # Log coverage statistics after imputation
    non_null_counts = mp_data.filter(like='modis_').notna().sum()
    logger.info("MODIS data coverage after imputation:")
    for col, count in non_null_counts.items():
        logger.info(f"{col}: {count}/{len(mp_data)} values ({count/len(mp_data)*100:.1f}%)")
    
    # Save merged dataset
    output_file = PATH.PROCESSED_DATA / "nettows_with_modis.parquet"
    mp_data.to_parquet(output_file)
    logger.info(f"Saved merged dataset to {output_file}")

if __name__ == "__main__":
    main()
