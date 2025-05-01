import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 1. Load and Aggregate Data
df = pd.read_csv('data/monthly_weather_summary.csv')

# Group by lat/lon and sum rainfall
rainfall_sum = df.groupby(['latitude', 'longitude'], as_index=False)['tp'].sum()
rainfall_sum.rename(columns={'tp': 'total_rainfall'}, inplace=True)

# 2. Prepare Data for Clustering
X = rainfall_sum[['latitude', 'longitude', 'total_rainfall']].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluate optimal k using elbow method and silhouette analysis
k_range = range(2, 31)  # Test k from 2 to 30
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.savefig('images/cluster_evaluation.png')
plt.close()

# Find optimal k based on silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k based on silhouette score: {optimal_k}")

# 3. Perform Clustering with optimal k
k = optimal_k  # Use the optimal k found
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

rainfall_sum['cluster'] = clusters

# 4. Analyze Clusters
cluster_stats = []

for cluster_id in range(k):
    cluster_points = rainfall_sum[rainfall_sum['cluster'] == cluster_id]
    
    # Cluster center in original units
    center_scaled = kmeans.cluster_centers_[cluster_id]
    center_scaled = center_scaled.reshape(1, -1)  # Reshape to 2D array
    center_original = scaler.inverse_transform(center_scaled)
    center_lat, center_lon, _ = center_original[0]  # Get first row since we reshaped to 2D

    distances = []
    for idx, row in cluster_points.iterrows():
        point = (row['latitude'], row['longitude'])
        center_point = (center_lat, center_lon)
        distance_km = geodesic(point, center_point).kilometers
        distances.append(distance_km)

    avg_distance = np.mean(distances)
    total_rainfall_cluster = cluster_points['total_rainfall'].sum()
    
    cluster_stats.append({
        'cluster_id': cluster_id,
        'avg_distance_to_center_km': avg_distance,
        'total_rainfall_cluster': total_rainfall_cluster
    })

# Convert to DataFrame
cluster_stats_df = pd.DataFrame(cluster_stats)

# 5. Output Results
print(cluster_stats_df)

# 6. Optional: Visualize
plt.figure(figsize=(10, 6))
plt.scatter(rainfall_sum['longitude'], rainfall_sum['latitude'], c=rainfall_sum['cluster'], cmap='tab10')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustering of Rainfall Points')
plt.colorbar(label='Cluster')
plt.savefig('images/cluster_map.png')
plt.close()
