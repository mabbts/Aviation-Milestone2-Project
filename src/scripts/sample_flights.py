from datetime import datetime
import pandas as pd
from pathlib import Path
from ..analysis.aggregations import get_sampled_flight_data

def main():
    # Configure sampling parameters
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    n_samples = 10
    output_dir = Path("data/sampled_flights")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sampled flight data
    print(f"Sampling {n_samples} dates between {start_date} and {end_date}")
    sampled_dfs = get_sampled_flight_data(start_date, end_date, n_samples)
    
    # Save results
    for i, df in enumerate(sampled_dfs):
        output_file = output_dir / f"flights_sample_{i+1}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved sample {i+1} to {output_file}")
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'sample_number': range(1, len(sampled_dfs) + 1),
        'num_flights': [len(df) for df in sampled_dfs],
        'file_name': [f"flights_sample_{i+1}.csv" for i in range(len(sampled_dfs))]
    })
    
    # Save summary
    summary.to_csv(output_dir / "sampling_summary.csv", index=False)
    print("\nSampling summary:")
    print(summary)

if __name__ == "__main__":
    main() 