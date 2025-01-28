# The Trino database

## `classpyopensky.trino.Trino(*args, **kwargs)`

Bases: `OpenSkyDBAPI`

Wrapper to OpenSky Trino database.

Credentials are fetched from the configuration file.

All methods return standard structures. When calls are made from the traffic library, they return advanced structures.

### `flightlist(start, stop=None, *args, departure_airport=None, arrival_airport=None, airport=None, callsign=None, icao24=None, cached=True, compress=False, limit=None, extra_columns=None, Table=<class 'pyopensky.schema.FlightsData4'>, **kwargs)`

Lists flights departing or arriving at a given airport.

You may pass requests based on time ranges, callsigns, aircraft, areas, serial numbers for receivers, or airports of departure or arrival.

The method builds appropriate SQL requests, caches results and formats data into a proper pandas DataFrame. Requests are split by hour (by default) in case the connection fails.

**Parameters:**

- **start** (*timelike*) – a string (default to UTC), epoch or datetime (native Python or pandas)
- **stop** (*None | timelike*) – a string (default to UTC), epoch or datetime (native Python or pandas), by default, one day after start

**Return type:**

`None | pd.DataFrame`

**More arguments to filter resulting data:**

**Parameters:**

- **departure\_airport** (*None | str | list[str]*) – a string for the ICAO identifier of the airport. Selects flights departing from the airport between the two timestamps;
- **arrival\_airport** (*None | str | list[str]*) – a string for the ICAO identifier of the airport. Selects flights arriving at the airport between the two timestamps;
- **airport** (*None | str | list[str]*) – a string for the ICAO identifier of the airport. Selects flights departing from or arriving at the airport between the two timestamps;
- **callsign** (*None | str | list[str]*) – a string or a list of strings (wildcards accepted, `_` for any character, `%` for any sequence of characters);
- **icao24** (*None | str | list[str]*) – a string or a list of strings identifying the transponder code of the aircraft;

**Warning**

If both `departure_airport` and `arrival_airport` are set, requested timestamps match the arrival time;

If `airport` is set, `departure_airport` and `arrival_airport` cannot be specified (a `RuntimeException` is raised).

**Useful options for debug**

**Parameters:**

- **cached** (*bool*) – (default: True) switch to `False` to force a new request to the database regardless of the cached files. This option also deletes previous cache files;
- **compress** (*bool*) – (default: False) compress cache files. Reduces disk space occupied at the expense of slightly increased time to load.
- **limit** (*None | int*) – maximum number of records requested, `LIMIT` keyword in SQL.

### `history(start, stop=None, *args, callsign=None, icao24=None, serials=None, bounds=None, departure_airport=None, arrival_airport=None, airport=None, time_buffer=None, cached=True, compress=False, limit=None, selected_columns=(), **kwargs)`

Get Traffic from the OpenSky Trino database.

You may pass requests based on time ranges, callsigns, aircraft, areas, serial numbers for receivers, or airports of departure or arrival.

The method builds appropriate SQL requests, caches results and formats data into a proper pandas DataFrame. Requests are split by hour (by default) in case the connection fails.

**Parameters:**

- **start** (*timelike*) – a string (default to UTC), epoch or datetime (native Python or pandas)
- **stop** (*None | timelike*) – a string (default to UTC), epoch or datetime (native Python or pandas), by default, one day after start
- **date\_delta** – a timedelta representing how to split the requests, by default: per hour

**Return type:**

`None | pd.DataFrame`

**More arguments to filter resulting data:**

**Parameters:**

- **callsign** (*None | str | list[str]*) – a string or a list of strings (wildcards accepted, `_` for any character, `%` for any sequence of characters);
- **icao24** (*None | str | list[str]*) – a string or a list of strings identifying the transponder code of the aircraft;
- **serials** (*None | int | Iterable[int]*) – an integer or a list of integers identifying the sensors receiving the data;
- **bounds** (*None | str | HasBounds | tuple[float, float, float, float]*) – sets a geographical footprint. Either an airspace or shapely shape (requires the `bounds` attribute); or a tuple of float (west, south, east, north);
- **selected\_columns** (*tuple[InstrumentedAttribute[Any] | str, …]*) – specify the columns you want to retrieve. When empty, use all columns of the `StateVectorsData4` table. You may escape column names as str. Always escape names from the `FlightsData4` table.

**Airports**

The following options build more complicated requests by merging information from two tables in the Trino database, resp. `state_vectors_data4` and `flights_data4`.

**Parameters:**

- **departure\_airport** (*None | str*) – a string for the ICAO identifier of the airport. Selects flights departing from the airport between the two timestamps;
- **arrival\_airport** (*None | str*) – a string for the ICAO identifier of the airport. Selects flights arriving at the airport between the two timestamps;
- **airport** (*None | str*) – a string for the ICAO identifier of the airport. Selects flights departing from or arriving at the airport between the two timestamps;
- **time\_buffer** (*None | str | pd.Timedelta*) – (default: None) time buffer used to extend time bounds for flights in the OpenSky flight tables: requests will get flights between `start` - `time_buffer` and `stop` + `time_buffer`. If no airport is specified, the parameter is ignored.

**Warning**

See `pyopensky.trino.flightlist()` if you do not need any trajectory information.

If both `departure_airport` and `arrival_airport` are set, requested timestamps match the arrival time;

If `airport` is set, `departure_airport` and `arrival_airport` cannot be specified (a `RuntimeException` is raised).

**Useful options for debug**

**Parameters:**

- **cached** (*bool*) – (default: True) switch to `False` to force a new request to the database regardless of the cached files. This option also deletes previous cache files;
- **compress** (*bool*) – (default: False) compress cache files. Reduces disk space occupied at the expense of slightly increased time to load.
- **limit** (*None | int*) – maximum number of records requested, `LIMIT` keyword in SQL.

### `rawdata(start, stop=None, *args, icao24=None, serials=None, bounds=None, callsign=None, departure_airport=None, arrival_airport=None, airport=None, cached=True, compress=False, limit=None, Table=<class 'pyopensky.schema.RollcallRepliesData4'>, extra_columns=(), **kwargs)`

Get raw message from the OpenSky Trino database.

You may pass requests based on time ranges, callsigns, aircraft, areas, serial numbers for receivers, or airports of departure or arrival.

The method builds appropriate SQL requests, caches results and formats data into a proper pandas DataFrame. Requests are split by hour (by default) in case the connection fails.

**Parameters:**

- **start** (*timelike*) – a string (default to UTC), epoch or datetime (native Python or pandas)
- **stop** (*None | timelike*) – a string (default to UTC), epoch or datetime (native Python or pandas), by default, one day after start
- **date\_delta** – a timedelta representing how to split the requests, by default: per hour

**Return type:**

`None | pd.DataFrame`

**More arguments to filter resulting data:**

**Parameters:**

- **callsign** (*None | str | list[str]*) – a string or a list of strings (wildcards accepted, `_` for any character, `%` for any sequence of characters);
- **icao24** (*None | str | list[str]*) – a string or a list of strings identifying the transponder code of the aircraft;
- **serials** (*None | int | Iterable[int]*) – an integer or a list of integers identifying the sensors receiving the data;
- **bounds** (*None | HasBounds | tuple[float, float, float, float]*) – sets a geographical footprint. Either an airspace or shapely shape (requires the `bounds` attribute); or a tuple of float (west, south, east, north);

**Airports**

The following options build more complicated requests by merging information from two tables in the Trino database, resp. `rollcall_replies_data4` and `flights_data4`.

**Parameters:**

- **departure\_airport** (*None | str*) – a string for the ICAO identifier of the airport. Selects flights departing from the airport between the two timestamps;
- **arrival\_airport** (*None | str*) – a string for the ICAO identifier of the airport. Selects flights arriving at the airport between the two timestamps;
- **airport** (*None | str*) – a string for the ICAO identifier of the airport. Selects flights departing from or arriving at the airport between the two timestamps;

**Warning**

If both `departure_airport` and `arrival_airport` are set, requested timestamps match the arrival time;

If `airport` is set, `departure_airport` and `arrival_airport` cannot be specified (a `RuntimeException` is raised).

It is not possible at the moment to filter both on airports and on geographical bounds (help welcome!).

**Useful options for debug**

**Parameters:**

- **cached** (*bool*) – (default: True) switch to `False` to force a new request to the database regardless of the cached files. This option also deletes previous cache files;
- **compress** (*bool*) – (default: False) compress cache files. Reduces disk space occupied at the expense of slightly increased time to load.
- **limit** (*None | int*) – maximum number of records requested, `LIMIT` keyword in SQL.