# About This Fork

This project was completed as part of a team milestone in the University of Michigan's Master of Applied Data Science program. My contributions include data acquisition, validating and contributing to various code elements, report writing, and subject matter expertise regarding the ADBS sytem.

 **Final Report (`Milestone II – Project Report (1).docx`)**

# Flight Data Analysis and Prediction System

This project is a comprehensive system for retrieving, processing, analyzing, and predicting aircraft flight data from the **OpenSky Network**. It consists of three main components:

1. **Data Pipeline (`src/`)**: A robust pipeline for retrieving and processing flight data
2. **State Prediction (`state_prediction/`)**: Advanced machine learning models for predicting future aircraft states
3. **Flight Classification (`enid/`)**: Unsupervised learning techniques for classifying flight patterns

## Table of Contents
- [Project Overview](#project-overview)
- [Key Components](#key-components)
  - [Data Pipeline (src)](#data-pipeline-src)
  - [State Prediction](#state-prediction)
  - [Flight Classification (ENID)](#flight-classification-enid)
- [Directory Structure](#directory-structure)
- [Example Usage](#example-usage)
- [Current Progress](#current-progress)
- [Next Steps](#next-steps)
- [Contributing](#contributing)

---

## Project Overview

This repository leverages the **OpenSky Network** ADS-B data to:

- Ingest and store flight-related data (flights, state vectors)
- Preprocess raw flight tracks for analysis and modeling
- Predict future aircraft states using various machine learning approaches
- Classify flight patterns using unsupervised learning techniques

We aim to build scalable data pipelines and conduct robust analyses that can uncover patterns, support supervised/unsupervised learning, and demonstrate best practices in big data handling.

---

## Key Components

### Data Pipeline (src)

The data pipeline provides a structured framework for retrieving and processing flight data from the OpenSky Network.

#### Key Features:
- **Retrieval Engine**: Fetches flight data using time intervals and query functions
- **Pipeline Modules**: Specialized pipelines for different data types:
  - `FlightsPipeline`: Retrieves flight data with metadata
  - `StateVectorPipeline`: Retrieves state vector data (position, velocity, etc.)
- **Query Modules**: SQL query generators for different data types
- **Transformation Modules**: Data preprocessing utilities

### State Prediction

The state prediction component uses machine learning to predict future aircraft states based on historical trajectory data.

#### Models Implemented:
1. **Transformer**: Attention-based sequence model for capturing complex temporal dependencies
2. **LSTM**: Long Short-Term Memory network for sequential data
3. **FFNN**: Feed-Forward Neural Network for simpler prediction tasks
4. **XGBoost**: Gradient boosting for tabular data with engineered features
5. **Kalman Filter**: Traditional state estimation approach

#### Key Features:
- Model training and evaluation scripts
- Hyperparameter optimization
- Comprehensive metrics and visualization tools
- Prediction capabilities for single flights or batches
- Analysis tools for model performance and failure cases

### Flight Classification (ENID)

The flight classification component uses unsupervised learning techniques to identify and categorize flight patterns.

#### Key Techniques:
1. **Dynamic Time Warping (DTW)**: Algorithm for measuring similarity between temporal sequences
2. **K-means Clustering**: Unsupervised clustering to identify natural groupings of flight patterns
3. **Prototype Matching**: Comparison of flight patterns against predefined prototypes

#### Features:
- Support for multi-dimensional DTW to compare multiple flight attributes
- Prototype-based classification for known flight categories
- CNN-based classification after unsupervised labeling

---

## Directory Structure

Below is a high-level overview of the repository layout:

```
.
├── src/                           # Data pipeline code
│   ├── pipeline/                  # Data retrieval pipelines
│   │   ├── flight_pipeline.py     # Flight data pipeline
│   │   └── state_vector_pipeline.py # State vector pipeline
│   ├── queries/                   # SQL query generators
│   │   ├── flight_queries.py      # Flight data queries
│   │   └── state_vector_queries.py # State vector queries
│   ├── retrieval/                 # Data retrieval utilities
│   │   ├── interval_generation.py # Time interval generators
│   │   └── retrieval_engine.py    # Generic retrieval engine
│   ├── transformations/           # Data transformation utilities
│   │   ├── flight_preprocessing.py # Flight data preprocessing
│   │   └── state_preprocessing.py # State vector preprocessing
│   └── utils/                     # Utility functions
│       ├── constants.py           # Constant definitions
│       ├── file_utils.py          # File handling utilities
│       ├── paths.py               # Path definitions
│       └── time_utils.py          # Time handling utilities
│
├── state_prediction/              # State prediction models
│   ├── models.py                  # Model definitions
│   └── scripts/                   # Training and evaluation scripts
│       ├── analyze_model_failures.py # Failure analysis
│       ├── config.py              # Configuration
│       ├── evaluate.py            # Model evaluation
│       ├── feature_analysis.py    # Feature importance analysis
│       ├── forecast_single_flight.py # Single flight prediction
│       ├── paths.py               # Path definitions
│       ├── predict.py             # Prediction script
│       ├── prepare_data.py        # Data preparation
│       ├── sensitivity_analysis.py # Hyperparameter sensitivity
│       ├── train.py               # Model training
│       ├── train_xgboost.py       # XGBoost training
│       └── visualize_flights.py   # Visualization utilities
│
├── enid/                          # Flight classification
│   └── flight_classification/     # Classification utilities
│       └── cnn_experimentation2.ipynb # Prototype matching and CNN
│
├── data/                          # Data storage (not in repo)
│   ├── raw/                       # Raw data from OpenSky
│   └── processed/                 # Processed data for modeling
│
├── setup.py                       # Setup script
└── README.md                      # This file
└── Milestone II – Project Report (1).docx # Final report for the project (view as raw)
```

---

## Example Usage

### Data Pipeline

```python
from src.pipeline.flight_pipeline import FlightsPipeline
from src.utils.paths import DATA_DIR

# Retrieve flight data in 1-hour chunks
FlightsPipeline.chunked_flight_v4(
    start_date="2023-01-01",
    end_date="2023-01-02",
    output_dir=DATA_DIR / "flights",
    chunk_hours=1.0
)
```

### State Prediction

```bash
# Train a transformer model
python state_prediction/scripts/train.py --model transformer

# Evaluate model performance
python state_prediction/scripts/evaluate.py --model transformer --k_folds 5

# Generate predictions for a flight
python state_prediction/scripts/forecast_single_flight.py --model transformer --flight_file path/to/flight.parquet
```

### Flight Classification

The flight classification component is primarily implemented in Jupyter notebooks for exploratory analysis and visualization. The main notebook is `enid/flight_classification/cnn_experimentation2.ipynb`.

---

## Current Progress

- **Data Pipeline**: Fully implemented and tested for both flight data and state vector data
- **State Prediction**: 
  - Implemented multiple model architectures (Transformer, LSTM, XGBoost)
  - Created comprehensive evaluation framework
  - Analyzed model performance and failure cases
- **Flight Classification**:
  - Implemented DTW-based similarity measurement
  - Created K-means clustering for flight pattern identification
  - Developed prototype matching for classification
  - Explored CNN-based classification after unsupervised labeling

---

## Next Steps

- **Data Pipeline**:
  - Support for additional data sources
  - Real-time data streaming capabilities
- **State Prediction**:
  - Multi-modal prediction incorporating weather data
  - Uncertainty quantification in predictions
  - Ensemble methods combining multiple model types
- **Flight Classification**:
  - Integration of supervised learning with domain expert labels
  - Anomaly detection for unusual flight patterns
  - Real-time classification capabilities

---

## Contributing

This project is part of a course requirement, but feedback, suggestions, and ideas are welcome! Feel free to open issues or submit pull requests if you have improvements to suggest.

- **Issues**: For bug reports or feature requests
- **Pull Requests**: We welcome code contributions—please be sure to include clear descriptions and testing steps

