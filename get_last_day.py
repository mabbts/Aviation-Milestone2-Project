from data_loader import OpenSkyLoader

# Usage Example
# - snapshot of aircraft activity around Atlanta airport within a specific hour on March 1st, 2024, 
# focusing on flights related to that airport and capturing their state vector data
#  within a slightly extended timeframe of their observed flight duration.
if __name__ == "__main__":
    loader = OpenSkyLoader(
        config_path='config.yaml', 
        default_region='georgia',
        time_buffer='5 minutes'  # Buffer for finding flights
    )
    
    try:
        # Get flights data for Atlanta airport
        print("\nGetting flight data...")
        flight_data = loader.get_flight_data(
            start_time='2024-03-01 08:00:00',
            end_time='2024-03-01 09:00:00',
            airport='KATL',  # Atlanta International
            region='georgia'
        )
        
        if not flight_data.empty:
            print(f"\nFound {len(flight_data)} flight records")
            print("\nSample flight data:")
            print(flight_data[['time', 'icao24', 'callsign', 'lat', 'lon', 'departure_airport', 'arrival_airport']].head())
            
    except Exception as e:
        print(f"Data loading failed: {str(e)}")