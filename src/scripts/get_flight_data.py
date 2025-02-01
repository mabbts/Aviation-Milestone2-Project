from ..analysis.aggregations import get_flight_data_chunks
from ..utils.paths import DATA_DIR

def main():
    # Define date range for testing (12 hour period)
    start_date = '2025-01-01'
    end_date = '2025-01-01 12:00:00'
    
    print(f"Fetching flight data from {start_date} to {end_date}...")

    # Get the aggregated flight data
    get_flight_data_chunks(start_date, end_date, output_dir=DATA_DIR / 'flight_data')

if __name__ == "__main__":
    main()