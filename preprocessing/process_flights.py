from src.utils.paths import DATA_DIR
import pandas as pd

# combine all flight data chunks into a single dataframe

def combine_flight_data_chunks():
    # get all flight data chunks
    flight_data_chunks = list(DATA_DIR.glob('flight_data/*.csv'))
    print(flight_data_chunks)

    # combine all chunks into a single dataframe
    flight_data = pd.concat([pd.read_csv(chunk) for chunk in flight_data_chunks])
    print(flight_data.head())

    # save combined dataframe to csv
    flight_data.to_csv(DATA_DIR / 'flight_data/flight_data.csv', index=False)

if __name__ == "__main__":
    combine_flight_data_chunks()