import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def apply_dbscan(df, eps=0.5, min_samples=5):
    # Convert lat/lon to radians (required for Haversine distance)
    coords = np.radians(df[['lat', 'long']].values)
    
    # Scale speed & altitude so they have comparable influence
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['speed', 'altitude']])
    
    # Combine lat/long with scaled speed & altitude
    features = np.hstack([coords, scaled_features])

    # Apply DBSCAN using Haversine for lat/long and Euclidean for speed/altitude
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
    
    # Add labels to DataFrame
    df['cluster'] = db.labels_
    return df


