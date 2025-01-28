from data_loader import OpenSkyLoader
from datetime import datetime, timedelta
import pandas as pd
import os

def get_monthly_samples(start_date, end_date, airport='KATL', samples_per_month=5):
    """
    Get flight samples for each month between start_date and end_date
    
    Args:
        start_date: Starting date for data collection
        end_date: Ending date for data collection
        airport: Airport ICAO code
        samples_per_month: Number of samples to collect per month
        
    Returns:
        Dictionary containing all flight and vector data, organized by month
    """
    loader = OpenSkyLoader()
    monthly_data = {}
    
    current_date = start_date
    while current_date < end_date:
        # Calculate the start and end of the current month
        month_start = current_date.replace(day=1, hour=0, minute=0, second=0)
        if current_date.month == 12:
            month_end = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            month_end = current_date.replace(month=current_date.month + 1, day=1)
        
        # Sample flights for this month
        try:
            month_key = current_date.strftime('%Y-%m')
            print(f"Collecting data for {month_key}...")
            
            sampled_data = loader.sample_flights_with_vectors(
                start_time=month_start.strftime('%Y-%m-%d %H:%M:%S'),
                end_time=month_end.strftime('%Y-%m-%d %H:%M:%S'),
                n_samples=samples_per_month,
                airport=airport
            )
            
            monthly_data[month_key] = sampled_data
            print(f"Successfully collected {len(sampled_data['flights'])} flights for {month_key}")
            
        except Exception as e:
            print(f"Error collecting data for {month_key}: {str(e)}")
        
        # Move to next month
        current_date = month_end
    
    return monthly_data

def ensure_data_dir(base_dir='data'):
    """Create data directory if it doesn't exist"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

# Calculate dates for the past year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Ensure data directory exists
data_dir = ensure_data_dir()

# Collect the samples
print(f"Collecting flight samples from {start_date.date()} to {end_date.date()}")
monthly_samples = get_monthly_samples(start_date, end_date, samples_per_month=100)

# Save and print summary of collected data
for month, data in monthly_samples.items():
    print(f"\nMonth: {month}")
    print(f"Number of flights: {len(data['flights'])}")
    print(f"Number of state vectors: {len(data['state_vectors'])}")
    
    # Save flights data
    flights_file = os.path.join(data_dir, f'flights_{month}.csv')
    data['flights'].to_csv(flights_file, index=False)
    print(f"Saved flights to {flights_file}")
    
    # Save state vectors data
    vectors_file = os.path.join(data_dir, f'state_vectors_{month}.csv')
    data['state_vectors'].to_csv(vectors_file, index=False)
    print(f"Saved state vectors to {vectors_file}")

print("\nAll data has been saved to the 'data' directory")
