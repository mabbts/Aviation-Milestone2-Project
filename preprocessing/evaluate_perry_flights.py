import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("raw_state_vt")
df["time"] = pd.to_datetime(df["time"])

# Compute min/max altitude per callsign efficiently with groupby
altitude_summary = df.groupby("callsign").agg(
    min_altitude = ("geoaltitude", "min"),
    max_altitude = ("geoaltitude", "max"),
    start_time=("time", "min"),
    end_time = ("time", "max"),
)

# Rename columns for clarity
altitude_summary["flight_duration"] = (altitude_summary["end_time"] - altitude_summary["start_time"]).dt.total_seconds() / 60

altitude_summary = altitude_summary.drop(columns = ["start_time", "end_time"])

filtered_altitude_summary = altitude_summary[altitude_summary["min_altitude"]< 1500]

# Plot min/max altitude per callsign
filtered_altitude_summary.plot(kind="bar", figsize=(12, 6), title="Min/Max Altitude per Flight")
plt.xlabel("Callsign")
plt.ylabel("Altitude (ft)")
plt.show()

