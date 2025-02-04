from ..src.analysis.aggregations import get_flight_aggregates

def main():
    # Define date range
    start_date = "2025-01-01"
    end_date = "2025-01-02"
    
    # Get the aggregated flight data
    df = get_flight_aggregates(start_date, end_date)
    
    # Display the results
    print(f"Flight aggregates from {start_date} to {end_date}:")
    print(df)
    
    # Optionally save to CSV
    df.to_csv("../../flight_aggregates_jan2025.csv", index=False)

if __name__ == "__main__":
    main()