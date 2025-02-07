# Milestone II: Big Data Project (SIADS 696)

Welcome to the **Milestone II** repository for the SIADS 696 course at the University of Michigan. This project explores **Supervised Learning** and **Unsupervised Learning** techniques on aviation data from the **OpenSky Network** using a pipeline-oriented approach.

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Directory Structure](#directory-structure)
- [Key Components](#key-components)
  - [Pipelines](#pipelines)
  - [Queries](#queries)
  - [Retrieval](#retrieval)
  - [Transformations](#transformations)
  - [Utilities](#utilities)
- [Scripts](#scripts)
- [Example Usage](#example-usage)
- [Current Progress](#current-progress)
- [Next Steps](#next-steps)
- [Contributing](#contributing)

---

## Project Overview

This repository leverages the **OpenSky Network** ADS-B data to:

- Ingest and store flight-related data (flights, state vectors).
- Preprocess raw flight tracks for later analysis or modeling.
- Provide an end-to-end pipeline for data retrieval, transformation, and storage.

We aim to build scalable data pipelines and conduct robust analyses that can uncover patterns, support supervised/unsupervised learning, and demonstrate best practices in big data handling.

---

## Objectives

1. **Data Retrieval & Storage**  
   Query data from the OpenSky Trino database in time-chunked or sampled intervals, storing results locally (Parquet/CSV).

2. **Data Preprocessing**  
   Clean, transform, and compute aggregate metrics on flight tracks for downstream modeling and analysis.

3. **Scalability**  
   Ensure solutions can handle large volumes of data efficiently by leveraging chunked retrieval and batch processing.

4. **Analysis & Modeling**  
   Lay groundwork for exploratory analysis, predictive modeling (supervised), and anomaly/clustering tasks (unsupervised).

5. **Reproducibility**  
   Provide clear scripts and modular code so the entire process (from data retrieval to transformation) can be easily replicated.

---

## Directory Structure

Below is a high-level overview of the repository layout:

```
.
├── scripts/
│   ├── get_flight_data.py
│   ├── get_state_vector_data.py
│   ├── sample_flights.py
│   ├── sample_state_vectors.py
│   ├── preprocess_flights.py
│   └── __init__.py
├── src/
│   ├── pipeline/
│   │   ├── flight_pipeline.py
│   │   └── state_vector_pipeline.py
│   ├── queries/
│   │   ├── flight_queries.py
│   │   └── state_vector_queries.py
│   ├── retrieval/
│   │   ├── retrieval_engine.py
│   │   └── interval_generation.py
│   ├── transformations/
│   │   └── flight_preprocessing.py
│   └── utils/
│       ├── constants.py
│       ├── file_utils.py
│       ├── paths.py
│       ├── time_utils.py
│       ├── utils.py
│       └── __init__.py
├── data/
│   ├── raw/
│   │   ├── v4_chunks/
│   │   ├── v4_samples/
│   │   └── state_vectors_chunks/
│   └── processed/
└── README.md
```

- **scripts**: Python entry-point scripts for retrieving or processing data.
- **src**:  
  - **pipeline**: High-level workflow pipelines (e.g., retrieving chunked or sampled flights, state vectors).  
  - **queries**: SQL query builders for OpenSky’s Trino database.  
  - **retrieval**: Core retrieval engine and interval generation (chunked or random sampling).  
  - **transformations**: Data transformation and preprocessing logic (e.g., computing flight track metrics).  
  - **utils**: Shared utilities (paths, file handling, constants, date/time parsing, etc.).

- **data**: Suggested folder structure for storing **raw** downloaded Parquet files and any **processed** or **analyzed** output.

---

## Key Components

### Pipelines

- **`flight_pipeline.py`**  
  Contains methods to retrieve flight data (v4) from the OpenSky database in either:
  1. **Chunked** intervals (`chunked_flight_v4`)
  2. **Random sampled** intervals (`sample_flight_v4`)

- **`state_vector_pipeline.py`**  
  Defines methods to retrieve state vector data:
  1. **Chunked** intervals (`chunked_state_vectors`)
  2. **Random sampled** intervals (`sample_state_vectors`)

These pipelines rely on **interval generation** (e.g., `generate_chunk_intervals`, `generate_sample_intervals`) and the **retrieval engine** to execute queries and save results.

### Queries

- **`flight_queries.py`**  
  Provides SQL query templates (`get_flight_data_v4`, `get_flight_data_v5`) to fetch flight data from different OpenSky datasets.

- **`state_vector_queries.py`**  
  Offers SQL to retrieve state vector data with optional geographic bounding (e.g., `GEORGIA_BOUNDS`).

### Retrieval

- **`retrieval_engine.py`**  
  Houses the `retrieve_data_by_intervals` function:
  - Executes SQL queries against OpenSky (via `pyopensky.trino.Trino`)
  - Saves output to Parquet using utility functions.

- **`interval_generation.py`**  
  - **`generate_chunk_intervals`**: Slices a time range into consecutive (e.g., hourly) intervals.  
  - **`generate_sample_intervals`**: Creates random intervals of a specified length between a start/end date.

### Transformations

- **`flight_preprocessing.py`**  
  - **`compute_track_metrics`**: Given a flight track, computes min/max/mean of latitude, longitude, altitude, heading, and on-ground percentage.  
  - **`preprocess_flight_data`**: Applies `compute_track_metrics` to a DataFrame’s `track` column, returns new columns (e.g., `time_min`, `time_max`, etc.).

### Utilities

- **`constants.py`**: Contains bounds for Georgia, airport mappings, and other shared constants.  
- **`time_utils.py`, `utils.py`**: Date/time parsing, random sampling of dates, and general helper functions.  
- **`file_utils.py`**: Logic for saving data safely to CSV/Parquet (skipping if file exists, etc.).  
- **`paths.py`**: Defines project directories (e.g., `DATA_DIR`) and ensures they exist.

---

## Scripts

1. **`get_flight_data.py`**  
   - Demonstrates how to invoke `FlightsPipeline.chunked_flight_v4` to pull flight data in hourly chunks.

2. **`sample_flights.py`**  
   - Shows usage of `FlightsPipeline.sample_flight_v4` to randomly sample intervals within a given date range.

3. **`get_state_vector_data.py`**  
   - Illustrates `StateVectorPipeline.chunked_state_vectors` to fetch state vector data in consecutive time slices.

4. **`sample_state_vectors.py`**  
   - Uses `StateVectorPipeline.sample_state_vectors` to retrieve random daily intervals of state vector data.

5. **`preprocess_flights.py`**  
   - Reads multiple Parquet files from a folder, concatenates them, applies `preprocess_flight_data`, and saves the result.

Each script can be run directly (e.g., `python scripts/get_flight_data.py`) once dependencies are installed and the environment is set up.

---

## Example Usage

1. **Install** dependencies (e.g., from `requirements.txt` or your environment file):
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure** your environment with credentials/permissions to query the OpenSky Trino endpoint.

3. **Retrieve chunked flights**:
   ```bash
   python scripts/get_flight_data.py
   ```
   - This will generate hourly Parquet files in `data/raw/v4_chunks`.

4. **Sample flight data**:
   ```bash
   python scripts/sample_flights.py
   ```
   - Creates random 1-hour intervals over a given date range, saving results in `data/raw/v4_samples`.

5. **Preprocess flights**:
   ```bash
   python scripts/preprocess_flights.py
   ```
   - Reads the downloaded Parquet files in `data/raw/v4_samples`, applies transformations, and saves them in `data/processed`.

---

## Current Progress

- **Chunked & Sampled Retrieval**: Implemented and tested for both flight data (v4) and state vector data, saving directly to Parquet.
- **Basic Preprocessing**: Functions to compute track metrics (min/max/mean alt/lon/lat, on-ground fraction, etc.) have been added.
- **Scripts**: Example scripts illustrating how to retrieve and preprocess data are operational.
- **Organized Codebase**: A modular structure (`pipeline`, `queries`, `retrieval`, `transformations`, `utils`, `scripts`) for clarity and scalability.

---

## Next Steps

- **Enhanced Preprocessing**: Improve the flight preprocessor to handle edge cases (e.g., incomplete tracks, merges).
- **Exploratory Data Analysis**: Generate visual summaries (flight density maps, altitude distributions, etc.).
- **Model Prototypes**: Implement initial supervised/unsupervised models (e.g., cluster flight patterns, predict flight delays).
- **Automation**: Integrate scheduling tools (e.g., Airflow or Cron) for routine data pulls.

---

## Contributing

This project is part of a course requirement, but feedback, suggestions, and ideas are welcome! Feel free to open issues or submit pull requests if you have improvements to suggest.

- **Issues**: For bug reports or feature requests.
- **Pull Requests**: We welcome code contributions—please be sure to include clear descriptions and testing steps.

Thank you for checking out the **SIADS 696 Big Data Project**. We hope our approach to modular, scalable data pipelines inspires your own data engineering and analytics projects!
