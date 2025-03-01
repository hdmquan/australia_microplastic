import os
from dotenv import load_dotenv
import pandas as pd
from loguru import logger
from modis_tools.granule_handler import GranuleHandler
from modis_tools.auth import ModisSession
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from cmr import GranuleQuery

from src.utils import PATH

def get_monthly_locations(df):
    """
    Group locations by month and return a dictionary of unique dates and locations
    """
    # Create month column for grouping
    df['Month'] = pd.to_datetime(df['DateUTC']).dt.to_period('M')
    
    # Group by month and get unique locations
    monthly_data = {}
    for month, group in df.groupby('Month'):
        monthly_data[month] = {
            'locations': group[['Latitude', 'Longitude']].drop_duplicates().values.tolist(),
            'date': month.to_timestamp()  # Convert Period to timestamp for API query
        }
    
    return monthly_data

def get_bbox_from_locations(locations, buffer=0.5):
    """Create a bounding box that encompasses all locations with a buffer"""
    lats, lons = zip(*locations)
    return [
        min(lons) - buffer,  # min longitude with buffer
        min(lats) - buffer,  # min latitude with buffer
        max(lons) + buffer,  # max longitude with buffer
        max(lats) + buffer   # max latitude with buffer
    ]

def download_modis_data(monthly_data):
    """
    Download MODIS data using modis_tools for each month
    monthly_data: Dictionary with months as keys and locations/date info as values
    """
    # Load environment variables and create session with credentials
    load_dotenv()
    username = os.getenv('EARTHDATA_USERNAME')
    password = os.getenv('EARTHDATA_PASSWORD')
    
    if not username or not password:
        logger.error("EARTHDATA_USERNAME or EARTHDATA_PASSWORD not set in environment variables")
        return
    
    session = ModisSession(username=username, password=password)
    handler = GranuleHandler()
    
    # Create MODIS directory if it doesn't exist
    modis_dir = PATH.DATA / "external" / "modis"
    modis_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_size = 0
    max_size = 1024 * 1024 * 1024  # 1GB in bytes
    
    # Also download for specific time periods for Sydney Harbour inference
    sydney_periods = [
        (datetime(2012, 5, 1), datetime(2012, 9, 30)),
        (datetime(2022, 5, 1), datetime(2022, 9, 30))
    ]
    
    # Sydney Harbour coordinates
    sydney_harbour = [(-33.8568, 151.2153)]
    
    # Process monthly data first
    for month, data in monthly_data.items():
        if downloaded_size > max_size:
            logger.warning(f"Reached size limit of 1GB. Stopping downloads.")
            break
            
        logger.info(f"Processing month: {month}")
        
        # Get bounding box for all locations in this month
        bbox = get_bbox_from_locations(data['locations'])
        date = data['date']
        
        # Query granules for the entire month
        start_date = date.strftime("%Y-%m-01")
        end_date = (date + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        
        logger.info(f"Querying for period: {start_date} to {end_date}, bbox: {bbox}")
        
        api = GranuleQuery()
        granules = api.short_name("MYD09GA")\
                     .version("061")\
                     .temporal(start_date, end_date)\
                     .bounding_box(*bbox)\
                     .get(limit=100)  # Increase limit to get more results
        
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
                logger.info(f"File already exists: {filename}")
                downloaded_size += file_path.stat().st_size
                continue
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error getting filename: {e}")
            continue
        
        # Download the best granule
        try:
            logger.info(f"Downloading granule: {best_granule.id}")
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
            
        except Exception as e:
            logger.error(f"Failed to download granule for {month}: {str(e)}")
            continue
    
    # Now process Sydney Harbour periods
    for start_date, end_date in sydney_periods:
        if downloaded_size > max_size:
            logger.warning(f"Reached size limit of 1GB. Stopping downloads.")
            break
        
        logger.info(f"Processing Sydney Harbour period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get bounding box for Sydney Harbour with a larger buffer
        bbox = get_bbox_from_locations(sydney_harbour, buffer=2.0)
        
        # Process each month in the period
        current_date = start_date
        while current_date <= end_date:
            month_start = current_date.replace(day=1)
            if current_date.month == 12:
                month_end = datetime(current_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
            
            month_end = min(month_end, end_date)  # Don't go past the end date
            
            logger.info(f"Querying for Sydney period: {month_start.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}")
            
            api = GranuleQuery()
            granules = api.short_name("MYD09GA")\
                         .version("061")\
                         .temporal(month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))\
                         .bounding_box(*bbox)\
                         .get(limit=100)
            
            if not granules:
                logger.warning(f"No granules found for Sydney period {month_start.strftime('%Y-%m')}")
                current_date = month_end + timedelta(days=1)
                continue
            
            logger.info(f"Found {len(granules)} granule(s) for Sydney period {month_start.strftime('%Y-%m')}")
            
            # Sort granules by cloud cover (if available) and coverage
            from modis_tools.models import Granule
            granule_objects = [Granule.parse_obj(g) for g in granules]
            
            # Download all granules for Sydney (we want more temporal coverage)
            for granule in granule_objects[:10]:  # Limit to 10 per month to avoid excessive downloads
                try:
                    # Check if file already exists
                    filename = Path(granule.links[0].href).name
                    file_path = modis_dir / filename
                    if file_path.exists():
                        logger.info(f"File already exists: {filename}")
                        continue
                    
                    logger.info(f"Downloading Sydney granule: {granule.id}")
                    new_files = handler.download_from_granules(
                        [granule],
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
                    
                    if downloaded_size > max_size:
                        logger.warning(f"Reached size limit of 1GB. Stopping downloads.")
                        return
                        
                except Exception as e:
                    logger.error(f"Failed to download Sydney granule: {str(e)}")
                    continue
            
            # Move to next month
            current_date = month_end + timedelta(days=1)

def main():
    # Read the processed data
    nettows = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")
    
    # Get monthly locations
    monthly_data = get_monthly_locations(nettows)
    
    logger.info(f"Found {len(monthly_data)} unique months to process")
    download_modis_data(monthly_data)

if __name__ == "__main__":
    main()