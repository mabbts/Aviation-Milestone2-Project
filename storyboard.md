# Aviation Data Analysis Project Storyboard

## Project Evolution

### 1. Data Loader Implementation
- Created `OpenSkyLoader` class to interface with OpenSky's Trino database
- Implemented core functionality for querying flights and state vectors
- Added configuration support for different geographical regions
- Implemented caching to optimize database queries

### 2. Flight Sampling Feature
- Added capability to sample random flights from specified time periods and airports
- Enhanced the loader to fetch corresponding state vectors for each sampled flight
- Fixed issues with column references in Trino queries
- Modified state vector columns to include essential flight data:
  - Time
  - Aircraft identifier (icao24)
  - Callsign
  - Position (latitude, longitude)
  - Flight parameters (velocity, heading, geoaltitude)

### 3. Code Improvements
- Added proper error handling and logging
- Implemented caching for better performance
- Added type hints for better code maintainability
- Configured .gitignore to exclude cache files and bytecode

## Next Steps
- [ ] Implement data visualization for flight paths
- [ ] Add statistical analysis of flight patterns
- [ ] Create sample analysis notebooks
- [ ] Add documentation for common use cases

## Technical Decisions
1. **Database Interface**: Using pyopensky's Trino interface for better query performance
2. **Data Caching**: Local caching implemented to reduce database load
3. **Error Handling**: Comprehensive error handling with informative messages
4. **Code Organization**: Separation of concerns between data loading and analysis
