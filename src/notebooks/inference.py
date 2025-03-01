import os
import pandas as pd
import numpy as np
import joblib
import rasterio
from rasterio.transform import from_origin
from datetime import datetime
from pathlib import Path
from loguru import logger
import pyhdf.SD as sd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import PolynomialFeatures

from src.utils import PATH
from src.utils.feature_eng import calculate_spectral_indices, add_polynomial_features

# Sydney Harbour coordinates
SYDNEY_HARBOUR_LAT = -33.8568
SYDNEY_HARBOUR_LON = 151.2153

def find_modis_files_for_period(start_date, end_date, modis_dir):
    """Find MODIS files for a given time period."""
    modis_files = list(modis_dir.glob("MYD09GA.A*.hdf"))
    
    if not modis_files:
        logger.error(f"No MODIS files found in directory: {modis_dir}")
        return []
    
    logger.info(f"Found {len(modis_files)} total MODIS files in {modis_dir}")
    
    matching_files = []
    for file in modis_files:
        try:
            year_doy = file.name.split('.')[1][1:]  # YYYYDDD
            file_date = datetime.strptime(year_doy, "%Y%j")
            
            if start_date <= file_date <= end_date:
                matching_files.append((file_date, file))
                logger.debug(f"Matched file: {file.name} for date {file_date.strftime('%Y-%m-%d')}")
            else:
                logger.debug(f"File outside period: {file.name} for date {file_date.strftime('%Y-%m-%d')}")
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse date from filename {file.name}: {e}")
            continue
    
    if not matching_files:
        logger.warning(f"No matching files found for period {start_date} to {end_date}")
        # List a few example filenames to help diagnose the issue
        if modis_files:
            logger.info("Example filenames in directory:")
            for file in modis_files[:5]:
                logger.info(f"  {file.name}")
        return []
    
    # Sort by date
    matching_files.sort()
    logger.info(f"Found {len(matching_files)} matching files for period {start_date} to {end_date}")
    return matching_files

def extract_modis_data(hdf_file):
    """Extract MODIS bands from HDF file and return as a dictionary."""
    hdf = sd.SD(str(hdf_file))
    
    try:
        datasets = hdf.datasets()
        bands_data = {}
        
        # Process bands 1-7 with suffix '_1'
        target_bands = [f'sur_refl_b0{i}_1' for i in range(1, 8)]
        
        for ds_name, ds_info in datasets.items():
            if ds_name in target_bands:
                try:
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
                    
                    # Store with original band name (without adding prefix)
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

def create_feature_dataframe(bands_data):
    """Create a dataframe with features for each pixel."""
    # Get dimensions from the first band
    first_band = next(iter(bands_data.values()))
    rows, cols = first_band.shape
    
    # Create arrays for pixel coordinates
    pixel_indices = np.indices((rows, cols))
    row_indices = pixel_indices[0].flatten()
    col_indices = pixel_indices[1].flatten()
    
    # Create dataframe
    df = pd.DataFrame({
        'row': row_indices,
        'col': col_indices
    })
    
    # Rename bands to match expected feature names
    band_mapping = {
        'sur_refl_b01_1': 'modis_sur_refl_b01_1',
        'sur_refl_b02_1': 'modis_sur_refl_b02_1',
        'sur_refl_b03_1': 'modis_sur_refl_b03_1',
        'sur_refl_b04_1': 'modis_sur_refl_b04_1',
        'sur_refl_b05_1': 'modis_sur_refl_b05_1',
        'sur_refl_b06_1': 'modis_sur_refl_b06_1',
        'sur_refl_b07_1': 'modis_sur_refl_b07_1'
    }
    
    # Add band values with correct names
    for band_name, band_data in bands_data.items():
        feature_name = band_mapping.get(band_name, band_name)
        df[feature_name] = band_data.flatten()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def add_polynomial_features(X, importance_df, top_n=3, update_features=False):
    """Add polynomial features to the input data."""
    # Define base feature columns - these should match what was used in training
    feature_cols = ['FDI', 'EVI', 'modis_sur_refl_b04_1']
    
    X = np.array(X)
    top_features = importance_df.nlargest(top_n, 'Importance')['Feature'].tolist()
    top_features_idx = [feature_cols.index(f) for f in top_features]
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X[:, top_features_idx])
    X_poly = X_poly[:, top_n:]  # Remove linear terms
    
    return np.hstack([X, X_poly])

def run_inference(model_info, df):
    """Run inference on the dataframe."""
    try:
        # Calculate spectral indices
        df = calculate_spectral_indices(df)
        
        # Comprehensive imputation for all columns
        # First, replace infinities with NaNs
        for col in df.columns:
            if df[col].dtype == float:
                # Replace infinities with NaNs first
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # For MODIS bands, use spatial interpolation for missing values
        modis_cols = [col for col in df.columns if col.startswith('modis_sur_refl_b')]
        for col in modis_cols:
            if df[col].isna().any():
                # For each missing value, try to use nearby valid pixels
                valid_mask = ~df[col].isna()
                if valid_mask.sum() > 10:  # Need enough valid points
                    try:
                        from scipy.interpolate import griddata
                        points = df.loc[valid_mask, ['row', 'col']].values
                        values = df.loc[valid_mask, col].values
                        grid_x, grid_y = df.loc[~valid_mask, ['row', 'col']].values.T
                        
                        # Interpolate missing values
                        interpolated = griddata(points, values, (grid_x, grid_y), method='nearest')
                        df.loc[~valid_mask, col] = interpolated
                        logger.info(f"Spatially interpolated {len(interpolated)} values for {col}")
                    except Exception as e:
                        logger.warning(f"Spatial interpolation failed for {col}: {e}")
        
        # For spectral indices, recalculate after band imputation
        df = calculate_spectral_indices(df)
        
        # Final pass to replace any remaining NaNs with column means or zeros
        for col in df.columns:
            if df[col].dtype == float and df[col].isna().any():
                nan_count = df[col].isna().sum()
                # Calculate mean of non-NaN values
                mean_val = df[col].mean()
                if np.isnan(mean_val):
                    df[col] = df[col].fillna(0)
                    logger.info(f"Filled {nan_count} NaNs with 0 in {col}")
                else:
                    df[col] = df[col].fillna(mean_val)
                    logger.info(f"Filled {nan_count} NaNs with mean ({mean_val:.4f}) in {col}")
        
        # Load feature importance info
        feature_importance_info = joblib.load(PATH.WEIGHTS / 'feature_importance_info.joblib')
        
        # Get original features (before polynomial transformation)
        base_features = feature_importance_info['original_features']
        
        # Ensure all required base columns exist
        missing_cols = [col for col in base_features if col not in df.columns]
        if missing_cols:
            # Try to fix column names if they don't have the 'modis_' prefix
            for col in missing_cols[:]:
                if col.startswith('modis_') and col.replace('modis_', '') in df.columns:
                    df[col] = df[col.replace('modis_', '')]
                    missing_cols.remove(col)
                elif 'sur_refl_b04_1' in df.columns and col == 'modis_sur_refl_b04_1':
                    df['modis_sur_refl_b04_1'] = df['sur_refl_b04_1']
                    missing_cols.remove(col)
        
        # Check again after attempted fixes
        missing_cols = [col for col in base_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required base columns: {missing_cols}")
        
        # Select base features
        X = df[base_features].values
        
        # Check for and handle any remaining infinities or NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Generate polynomial features manually
        top_n = feature_importance_info['top_n']
        top_features = feature_importance_info['importance_df'].nlargest(top_n, 'Importance')['Feature'].tolist()
        top_features_idx = [base_features.index(f) for f in top_features]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_top = X[:, top_features_idx]
        X_poly = poly.fit_transform(X_top)
        X_poly = X_poly[:, top_n:]  # Remove linear terms
        
        # Combine original features with polynomial features
        X_enhanced = np.hstack([X, X_poly])
        
        # Final check for any bad values
        X_enhanced = np.nan_to_num(X_enhanced, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract components from model info
        model = model_info['model']
        y_scaler = model_info['y_scaler']
        
        # Run inference and scale back to original values
        predictions_scaled = model.predict(X_enhanced)
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        return df
    
    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}")
        # Add more detailed error information
        if "infinity" in str(e) or "NaN" in str(e):
            # Check which columns have problematic values
            for col in df.columns:
                if df[col].dtype == float:
                    inf_count = np.isinf(df[col]).sum()
                    nan_count = np.isnan(df[col]).sum()
                    if inf_count > 0 or nan_count > 0:
                        logger.error(f"Column {col} has {inf_count} infinities and {nan_count} NaNs")
        raise

def save_as_tiff(df, output_path, rows, cols, date_str):
    """Save predictions as a GeoTIFF."""
    # Create an empty array filled with NaN
    prediction_array = np.full((rows, cols), np.nan)
    
    # Fill in the predictions
    for _, row in df.iterrows():
        i, j = int(row['row']), int(row['col'])
        prediction_array[i, j] = row['prediction']
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define geotransform (simplified - assumes standard MODIS projection)
    # For a proper implementation, you would extract the actual geotransform from the MODIS metadata
    transform = from_origin(SYDNEY_HARBOUR_LON - 5, SYDNEY_HARBOUR_LAT + 5, 0.01, 0.01)
    
    # Save as GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype=prediction_array.dtype,
        crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
        transform=transform,
    ) as dst:
        dst.write(prediction_array, 1)
    
    logger.info(f"Saved prediction to {output_path}")

def main():
    # Load the trained model info
    model_path = PATH.WEIGHTS / 'rf_feat_eng.joblib'
    logger.info(f"Loading model from {model_path}")
    model_info = joblib.load(model_path)
    
    # Define time periods
    periods = [
        (datetime(2012, 5, 1), datetime(2012, 9, 30)),
        (datetime(2022, 5, 1), datetime(2022, 9, 30))
    ]
    
    # Create results directory
    results_dir = PATH.RESULTS / "sydney_inference"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and process MODIS files
    modis_dir = PATH.DATA / "external" / "modis"
    
    # Track statistics
    total_files = 0
    skipped_files = 0
    processed_files = 0
    failed_files = 0
    
    for start_date, end_date in periods:
        period_str = f"{start_date.year}_{start_date.month:02d}_to_{end_date.month:02d}"
        logger.info(f"Processing period: {period_str}")
        
        # Find MODIS files for this period
        modis_files = find_modis_files_for_period(start_date, end_date, modis_dir)
        
        if not modis_files:
            logger.warning(f"No MODIS files found for period {period_str}")
            continue
        
        logger.info(f"Found {len(modis_files)} MODIS files for period {period_str}")
        total_files += len(modis_files)
        
        # Process each file
        for file_date, file_path in modis_files:
            date_str = file_date.strftime("%Y%m%d")
            
            # Check if result already exists
            output_path = results_dir / f"sydney_plastics_{period_str}_{date_str}.tif"
            if output_path.exists():
                logger.info(f"Result already exists for {date_str}, skipping: {output_path}")
                skipped_files += 1
                continue
                
            logger.info(f"Processing file: {file_path.name} for date {date_str}")
            
            try:
                # Extract MODIS data
                bands_data = extract_modis_data(file_path)
                
                if not bands_data:
                    logger.warning(f"No valid bands extracted from {file_path}")
                    failed_files += 1
                    continue
                
                # Get dimensions from the first band
                first_band = next(iter(bands_data.values()))
                rows, cols = first_band.shape
                
                # Create feature dataframe
                df = create_feature_dataframe(bands_data)
                
                if len(df) == 0:
                    logger.warning(f"No valid pixels found in {file_path}")
                    failed_files += 1
                    continue
                
                # Initialize df_with_predictions to None to avoid reference error
                df_with_predictions = None
                
                # Run inference
                try:
                    df_with_predictions = run_inference(model_info, df)
                except Exception as e:
                    logger.error(f"Inference failed: {str(e)}")
                    # Try to impute missing values if that's the issue
                    if 'Missing required columns' in str(e):
                        logger.info("Attempting to calculate missing spectral indices")
                        try:
                            # Ensure band names match what calculate_spectral_indices expects
                            for band_name in bands_data:
                                if band_name not in df.columns and f'modis_{band_name}' not in df.columns:
                                    df[f'modis_{band_name}'] = df[band_name] if band_name in df.columns else None
                            
                            df_with_predictions = run_inference(model_info, df)
                        except Exception as e2:
                            logger.error(f"Second inference attempt failed: {str(e2)}")
                            failed_files += 1
                            continue
                
                # Check if we have predictions before saving
                if df_with_predictions is None or 'prediction' not in df_with_predictions.columns:
                    logger.error("No predictions generated, skipping file")
                    failed_files += 1
                    continue
                
                # Save as TIFF
                save_as_tiff(df_with_predictions, output_path, rows, cols, date_str)
                processed_files += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                failed_files += 1
                continue
    
    # Log summary statistics
    logger.info(f"Inference complete. Summary:")
    logger.info(f"  Total files found: {total_files}")
    logger.info(f"  Files skipped (already processed): {skipped_files}")
    logger.info(f"  Files successfully processed: {processed_files}")
    logger.info(f"  Files failed: {failed_files}")

if __name__ == "__main__":
    main()
