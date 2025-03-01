import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_trajectory_features(df):
    """Extracts meaningful features from flight data."""
    features = []
    
    for flight_id, flight in df.groupby("flight_id"):
        lat_var = flight["lat"].var()
        long_var = flight["long"].var()
        alt_var = flight["altitude"].var()
        speed_var = flight["speed"].var()
        avg_speed = flight["speed"].mean()
        avg_altitude = flight["altitude"].mean()
        max_altitude = flight["altitude"].max()
        min_altitude = flight["altitude"].min()
        altitude_range = max_altitude - min_altitude

        # Compute heading changes
        flight = flight.copy()
        flight["heading"] = np.arctan2(flight["long"].diff(), flight["lat"].diff())
        flight["heading_change"] = flight["heading"].diff().abs()

        # Count alternating direction changes (grid pattern)
        alternating_changes = ((flight["heading_change"] > 0.2) & (flight["heading_change"].shift() > 0.2)).sum()

        features.append([
            flight_id, lat_var, long_var, alt_var, speed_var, avg_speed, avg_altitude, 
            altitude_range, alternating_changes
        ])

    return pd.DataFrame(features, columns=[
        "flight_id", "lat_var", "long_var", "alt_var", "speed_var", 
        "avg_speed", "avg_altitude", "altitude_range", "alternating_changes"
    ])

def pca_kmeans_clustering(df, n_clusters=5):
    """Applies K-Means clustering to categorize flights."""
    feature_df = extract_trajectory_features(df)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.iloc[:, 1:])

    # Apply PCA for dimensionality reduction (optional, but helps with visualization)
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_pca = pca.fit_transform(X_scaled)

    # Plot the PCA components to inspect how well the clusters are separated
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', label="Flight Paths")
    plt.title("PCA of Flight Path Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Assign categories based on new cluster centers
    cluster_mapping = create_cluster_mapping(kmeans, scaler, feature_df.columns[1:])
   
    # Visualizing the cluster results with PCA reduced dimensions
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f"Clusters (K={n_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    return [cluster_mapping[label] for label in labels]

# from sklearn.metrics import silhouette_score

# def pca_kmeans_clustering(df, n_clusters=5):
#     """Applies K-Means clustering and computes silhouette score."""
#     feature_df = extract_trajectory_features(df)

#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(feature_df.iloc[:, 1:])

#     # Apply PCA for visualization (optional)
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)

#     # Apply K-Means
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)

#     # Compute silhouette score
#     silhouette_avg = silhouette_score(X_scaled, labels)
#     print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.4f}")

#     # Visualizing PCA-reduced clusters
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
#     plt.title(f"Clusters (K={n_clusters}), Silhouette Score: {silhouette_avg:.2f}")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.show()

#     # Assign categories based on cluster centers
#     cluster_mapping = create_cluster_mapping(kmeans, scaler, feature_df.columns[1:])

#     return [cluster_mapping[label] for label in labels], silhouette_avg

def create_cluster_mapping(kmeans, scaler, feature_names):
    """Refines cluster labeling based on updated feature understanding."""
    centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Convert back to original scale
    df_centers = pd.DataFrame(centers, columns=feature_names)

    print("Cluster Centers:\n", df_centers)  # Debugging step

    mapping = {}

    for i, row in df_centers.iterrows():
        print(f"Cluster {i}: {row.to_dict()}")  # Inspect each cluster's features

        if row["avg_altitude"] > 10000 and row["avg_speed"] > 200:
            mapping[i] = "Commercial"
        elif row["avg_speed"] < 150 and row["altitude_range"] < 4000 and row["alternating_changes"] > 50:
            mapping[i] = "Surveillance"
        elif row["avg_altitude"] < 10000 and row["altitude_range"] > 5000 and row["alternating_changes"] > 50:
            mapping[i] = "Emergency"
        elif row["avg_speed"] > 150 and row["altitude_range"] > 3000 and row["alternating_changes"] < 10:
            mapping[i] = "Private"
        elif row["avg_speed"] < 200 and row["altitude_range"] < 5000 and row["alternating_changes"] > 100:
            mapping[i] = "Training"
        else:
            mapping[i] = "Other"

    print("Cluster Mapping:", mapping)  # Debugging step


    return mapping


