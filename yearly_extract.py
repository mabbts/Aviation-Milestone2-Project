from data_loader import OpenSkyLoader
from datetime import datetime, timedelta
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_data_dir(base_dir='data'):
    """Create data directory if it doesn't exist"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def process_month(year: int, month: int, airport: str = 'KATL', data_dir: str = 'data'):
    """
    Process and save data for a single month
    
    Args:
        year: Year to process
        month: Month to process (1-12)
        airport: Airport ICAO code
        data_dir: Directory to save data
    """
    # Calculate start and end dates
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    month_key = start_date.strftime('%Y-%m')
    logger.info(f"Processing {month_key}...")
    
    # Initialize loader and get data
    loader = OpenSkyLoader(request_delay=2.0)
    try:
        data = loader.get_flights_with_vectors(
            start_time=start_date,
            end_time=end_date,
            airport=airport,
            region='georgia'
        )
        
        if data['flights'].empty:
            logger.warning(f"No data found for {month_key}")
            return
        
        # Save the data
        flights_file = os.path.join(data_dir, f'flights_{month_key}.csv')
        vectors_file = os.path.join(data_dir, f'state_vectors_{month_key}.csv')
        
        data['flights'].to_csv(flights_file, index=False)
        data['state_vectors'].to_csv(vectors_file, index=False)
        
        logger.info(f"Saved {len(data['flights'])} flights and "
                   f"{len(data['state_vectors'])} vectors for {month_key}")
        
    except Exception as e:
        logger.error(f"Error processing {month_key}: {str(e)}")

if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = ensure_data_dir()
    
    # Process the last 12 months
    end_date = datetime.now()
    current_date = end_date - timedelta(days=30)
    
    # Round to start of current month
    current_date = current_date.replace(day=1, hour=0, minute=0, second=0)
    
    logger.info(f"Processing data from {current_date.date()} to {end_date.date()}")
    
    while current_date < end_date:
        process_month(current_date.year, current_date.month, data_dir=data_dir)
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    logger.info("Processing complete!")
