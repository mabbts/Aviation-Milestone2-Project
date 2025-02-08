# Aviation Data Analysis Project Storyboard

## Project Evolution

### 1. Data Loader Implementation
- **OpenSkyLoader Class** (initial approach): Implemented to interface with OpenSky’s Trino database.  
  - Provided core functionality for querying flights and state vectors.
  - Added configuration support for different geographical regions.
  - Implemented caching to optimize database queries.

### 2. Flight Sampling Feature (Initial)
- Added capability to sample random flights from specified time periods and airports.
- Enhanced the loader to fetch corresponding state vectors for each sampled flight.
- Fixed issues with column references in Trino queries.
- Modified state vector columns to include essential flight data:
  - Time, Aircraft identifier (icao24), Callsign
  - Position (latitude, longitude)
  - Flight parameters (velocity, heading, geoaltitude)

### 3. Code Improvements
- Added proper error handling and descriptive logging.
- Implemented caching for better performance and reduced load on the database.
- Introduced type hints to improve code clarity and maintainability.
- Updated `.gitignore` to exclude cache files and bytecode.

### 4. Pipeline Implementation for Chunked and Sample Retrieval
- **Pipeline Classes**:
  - **FlightsPipeline** (`flight_pipeline.py`):
    - `chunked_flight_v4(start_date, end_date, output_dir, ...)`: Retrieves flight data in fixed-size time chunks.
    - `sample_flight_v4(start_date, end_date, output_dir, n_samples, ...)`: Randomly samples time intervals for flight data retrieval.
  - **StateVectorPipeline** (`state_vector_pipeline.py`):
    - `chunked_state_vectors(start_date, end_date, output_dir, ...)`: Retrieves state vector data in fixed-size chunks.
    - `sample_state_vectors(start_date, end_date, output_dir, n_samples, ...)`: Randomly samples time intervals for state vector data.

- **Retrieval Engine** (`retrieval_engine.py`):
  - Implements a generic `retrieve_data_by_intervals(...)` function.
  - Uses a Trino client (`pyopensky.trino.Trino`) to execute SQL queries returned by your query functions.
  - Saves the retrieved data as parquet files to reduce repeated downloads.
  - Honors `skip_if_exists` to avoid overwriting existing data and speed up repeated runs.

- **Query Modules** (`flight_queries.py` and `state_vector_queries.py`):
  - Contain parameterized SQL query builders for different dataset tables (`flights_data4`, `flights_data5`, `state_vectors_data4`, etc.).
  - Support filtering by time range, airport, and geographical bounds.

- **Interval Generation** (`interval_generation.py`):
  - Functions to generate lists of `(start_timestamp, end_timestamp)` pairs:
    - `generate_chunk_intervals(...)` for uniform chunk-based retrieval.
    - `generate_sample_intervals(...)` for random sampling of intervals.

- **Scripts**:
  - `scripts/get_flight_data.py` and `scripts/get_state_vector_data.py`:  
    - Illustrate how to retrieve data in hourly chunks over a specified date range.
  - `scripts/sample_flights.py` and `scripts/sample_state_vectors.py`:  
    - Demonstrate how to sample data in random intervals.
  - `scripts/preprocess_flights.py`:  
    - Reads retrieved parquet files, applies preprocessing (e.g., computing track metrics), and writes processed files.

### 5. Data Transformations
- **Flight Preprocessing** (`flight_preprocessing.py`):
  - `compute_track_metrics(track)`: Aggregates min/max/mean for various parameters (time, latitude, longitude, altitude, heading) and calculates on-ground percentage.
  - `preprocess_flight_data(df)`: Applies `compute_track_metrics` to each row and merges the results as new columns.  

## Next Steps
- [ ] **Data Visualization**: Develop scripts or notebooks to visualize flight paths, altitudes, and other metrics over time.  
- [ ] **Statistical Analysis**: Incorporate advanced analytics (e.g., clustering, anomaly detection, or trend analyses) for flight patterns.  
- [ ] **Sample Analysis Notebooks**: Provide example Jupyter notebooks showing how to load and analyze the retrieved data.  
- [ ] **Documentation**: Expand and refine code documentation, including docstrings and a wiki for common use cases.  

## Technical Decisions
1. **Database Interface**: Using the `pyopensky.trino` library for efficient SQL querying against OpenSky data via Trino.  
2. **Pipeline Architecture**: Separating retrieval logic (queries, chunking/sampling) from transformations (preprocessing, aggregations).  
3. **Caching & Skipping**: Implementing file-based caching (`skip_if_exists`) ensures we only download new data, improving performance.  
4. **Flexible Queries**: Parameterized queries allow filtering by time range, airport, and geographic bounds to handle multiple data scenarios.  