from ..analysis.aggregations import get_detailed_flights

def main():
    # Define date range
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    
    # Get the aggregated flight data
    df = get_detailed_flights(start_date, end_date)
    
    # Display the results
    print(f"Flight details from {start_date} to {end_date}:")
    print(df)
    
    # Optionally save to CSV
    df.to_csv("../../flight_details_jan2024.csv", index=False)

if __name__ == "__main__":
    main()