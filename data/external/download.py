import os
from dotenv import load_dotenv
import pandas as pd
from loguru import logger
from modis_tools.granule_handler import GranuleHandler
from modis_tools.auth import ModisSession
from pathlib import Path
from datetime import datetime
import numpy as np
from cmr import GranuleQuery

from src.utils import PATH

def get_monthly_locations(df):
    """
    Group locations by month and return a dictionary of unique dates and locations
    """
    # Create month column for grouping
    df['Month'] = df['DateUTC'].dt.to_period('M')
    
    # Group by month and get unique locations
    monthly_data = {}
    for month, group in df.groupby('Month'):
        monthly_data[month] = {
            'locations': group[['Latitude', 'Longitude']].drop_duplicates().values.tolist(),
            'date': month.to_timestamp()  # Convert Period to timestamp for API query
        }
    
    return monthly_data

def get_bbox_from_locations(locations):
    """Create a bounding box that encompasses all locations"""
    lats, lons = zip(*locations)
    return [
        min(lons),  # min longitude
        min(lats),  # min latitude
        max(lons),  # max longitude
        max(lats)   # max latitude
    ]

def download_modis_data(monthly_data):
    """
    Download MODIS data using modis_tools for each month
    monthly_data: Dictionary with months as keys and locations/date info as values
    """
    # Load environment variables and create session with credentials
    load_dotenv()
    session = ModisSession(
        username=os.getenv('EARTHDATA_USERNAME'),
        password=os.getenv('EARTHDATA_PASSWORD')
    )
    
    handler = GranuleHandler()
    
    # Create MODIS directory if it doesn't exist
    modis_dir = PATH.DATA / "external"  / "modis"
    modis_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_size = 0
    max_size = 1024 * 1024 * 1024  # 1GB in bytes
    
    for month, data in monthly_data.items():
        if downloaded_size > max_size:
            logger.warning(f"Reached size limit of 1GB. Stopping downloads.")
            break
            
        logger.info(f"Processing month: {month}")
        
        # Get bounding box for all locations in this month
        bbox = get_bbox_from_locations(data['locations'])
        date = data['date']
        
        # Query granules for the entire month
        api = GranuleQuery()
        granules = api.short_name("MYD09GA")\
                     .version("061")\
                     .temporal(
                         date.strftime("%Y-%m-01"),
                         (date + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
                     )\
                     .bounding_box(*bbox)\
                     .get()
        
        if not granules:
            logger.warning(f"No granules found for month {month}")
            continue
        
        logger.info(f"Found {len(granules)} granule(s) for {month}")
        
        # Sort granules by cloud cover (if available) and coverage
        from modis_tools.models import Granule
        granule_objects = [Granule.parse_obj(g) for g in granules]
        
        # Select the best granule (least cloudy and most complete)
        best_granule = None
        min_cloud_cover = float('inf')
        
        for granule in granule_objects:
            try:
                # Check cloud cover from metadata if available
                # This is a simplified example - you might need to adjust based on actual metadata
                cloud_cover = float(granule.cloud_cover) if hasattr(granule, 'cloud_cover') else 100
                
                if cloud_cover < min_cloud_cover:
                    min_cloud_cover = cloud_cover
                    best_granule = granule
            except (ValueError, AttributeError):
                continue
        
        if not best_granule:
            best_granule = granule_objects[0]  # Take first if no cloud info available
        
        # Check if file already exists
        try:
            filename = Path(best_granule.links[0].href).name
            file_path = modis_dir / filename
            if file_path.exists():
                downloaded_size += file_path.stat().st_size
                logger.info(f"File already exists: {filename}")
                continue
        except (IndexError, AttributeError):
            continue
        
        # Download the best granule
        try:
            new_files = handler.download_from_granules(
                [best_granule],
                modis_session=session,
                path=modis_dir,
                threads=1,
                force=False
            )
            
            # Update size tracking
            for file_path in new_files:
                file_size = Path(file_path).stat().st_size
                downloaded_size += file_size
                logger.info(f"Downloaded {file_path.name} ({file_size / 1024 / 1024:.1f} MB)")
            
            logger.info(f"Total size so far: {downloaded_size / 1024 / 1024:.1f} MB")
            
        except FileNotFoundError as e:
            logger.error(f"Failed to download granule for {month}: {str(e)}")
            continue

def main():
    # Read the processed data
    nettows = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")
    
    # Get monthly locations
    monthly_data = get_monthly_locations(nettows)
    
    logger.info(f"Found {len(monthly_data)} unique months to process")
    download_modis_data(monthly_data)

if __name__ == "__main__":
    main()