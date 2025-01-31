# Milestone II: Big Data Project

Welcome to the **Milestone II** repository, a project for the SIADS 696 course at the University of Michigan. This project explores **Supervised Learning and Unsupervised Learning**, focusing on aviation data from the **OpenSky Network**. 

## Project Overview

The primary goal of this project is to leverage the OpenSky aviation ADS-B dataset to uncover patterns, analyze trends, and create predictive models. Our work aims to showcase the power of big data tools and methodologies to gain actionable insights from real-world data.

## Objectives

- **Data Exploration**: Analyze and preprocess the OpenSky ADS-B data to understand its structure and identify trends.
- **Big Data Processing**: Utilize scalable tools to handle large volumes of data efficiently.
- **Visualization and Reporting**: Create intuitive visualizations and summaries to present findings.
- **Predictive Modeling**: Develop models to forecast flight behaviors, anomalies, or other aviation metrics.
- **Reproducibility**: Document workflows and provide scripts to replicate the analysis and results.s

## Project Structure

```
.
├── src/
│ ├── analysis/ # Analysis and aggregation scripts
│ ├── database/ # Database queries and connections
│ ├── utils/ # Constants and utility functions
├── notebooks/ # Jupyter notebooks for exploration
├── tests/ # Test scripts
├── docs/ # Documentation
└── preprocessing/ # Data preprocessing scripts
```


## Key Components

### Database Queries
The `src/database/queries.py` module contains a collection of SQL queries for the OpenSky Trino database, including:
- Flight aggregation queries
- Flight path tracking
- Aircraft-specific queries

### Analysis Tools
Located in `src/analysis/`, including:
- Flight data aggregations
- Statistical analysis
- Pattern recognition

### Data Processing Scripts
The `preprocessing/` directory contains scripts for:
- Extracting flight data for Georgia
- Processing raw ADS-B data
- Data cleaning and transformation

### Configuration
Key constants and configurations in `src/utils/constants.py`:
- Georgia geographical bounds
- Airport definitions
- Time window configurations

## Features

- **Big Data Pipeline**: End-to-end pipeline for processing, analyzing, and visualizing large datasets.
- **Interactive Visualizations**: User-friendly visualizations to explore key insights.
- **Predictive Analytics**: Models built for specific predictions, such as flight path anomalies or delays.
- **Scalability**: Designed to handle large datasets efficiently using distributed computing.

## Current Progress

- Dataset exploration and initial preprocessing have been completed.
- The project framework is being developed with scalability and modularity in mind.
- Tools and libraries under consideration include PySpark, Pandas, and visualization frameworks.

## Next Steps

- Develop a robust preprocessing pipeline.
- Share early visualizations and insights.
- Continue refining the project scope and deliverables.

## Contributing

This project is part of an academic course, but external contributions, feedback, or ideas are welcome. 
