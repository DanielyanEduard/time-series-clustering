from visualization import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import DBSCAN
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from tslearn.metrics import dtw
from kneed import KneeLocator

# Load the Trace dataset
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

# Combine train and test for a larger dataset
X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

# Normalize the time series data
scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X)

# Plot original data by class
plot_original = plot_time_series_by_class(X_scaled, y, "Trace Dataset - True Classes")
plot_original.savefig('graphs/true_classes.png')

# Calculate DTW distance matrix (for hierarchical clustering and DBSCAN)
print("Calculating DTW distance matrix...")
n_samples = X_scaled.shape[0]
dtw_matrix = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    for j in range(i + 1, n_samples):
        dtw_dist = dtw(X_scaled[i].ravel(), X_scaled[j].ravel())
        dtw_matrix[i, j] = dtw_dist
        dtw_matrix[j, i] = dtw_dist

# ==========================================
# Method 1: K-means with DTW
# ==========================================
print("Performing K-means clustering with DTW...")
n_classes = len(np.unique(y))
k = n_classes  # Use the same number of clusters as true classes

# Apply K-means with DTW distance
km_dtw = TimeSeriesKMeans(n_clusters=k,
                          metric="dtw",
                          verbose=True,
                          random_state=42,
                          n_jobs=-1)
y_km_dtw = km_dtw.fit_predict(X_scaled)

# Calculate silhouette score for DTW K-means
sil_km_dtw = silhouette_score(dtw_matrix, y_km_dtw, metric='precomputed')
ari_km_dtw = adjusted_rand_score(y, y_km_dtw)

print(f"DTW K-means - Silhouette Score: {sil_km_dtw:.4f}")
print(f"DTW K-means - Adjusted Rand Index: {ari_km_dtw:.4f}")

# Plot K-means with DTW results
plot_km_dtw = plot_clustering_results(X_scaled, y_km_dtw, "DTW K-means Clustering Results")
plot_km_dtw.savefig('graphs/kmeans_dtw_clusters.png')

# ==========================================
# Method 2: Hierarchical Clustering with DTW
# ==========================================
print("Performing Hierarchical clustering with DTW...")

# Apply hierarchical clustering using the pre-computed DTW distance matrix
Z = linkage(dtw_matrix[np.triu_indices(n_samples, k=1)], method='ward')

# Cut the dendrogram to get the same number of clusters as the original classes
from scipy.cluster.hierarchy import fcluster

y_hc = fcluster(Z, t=k, criterion='maxclust') - 1  # Adjust to 0-based indexing

# Calculate silhouette score for Hierarchical clustering
sil_hc = silhouette_score(dtw_matrix, y_hc, metric='precomputed')
ari_hc = adjusted_rand_score(y, y_hc)

print(f"Hierarchical Clustering - Silhouette Score: {sil_hc:.4f}")
print(f"Hierarchical Clustering - Adjusted Rand Index: {ari_hc:.4f}")

# Plot hierarchical clustering results
plot_hc = plot_clustering_results(X_scaled, y_hc, "Hierarchical Clustering Results")
plot_hc.savefig('graphs/hierarchical_clusters.png')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=k * 2,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
)
plt.savefig('graphs/dendrogram.png')

# ==========================================
# Method 3: KShape Clustering
# ==========================================
print("Performing KShape clustering...")

# Apply KShape clustering (which uses shape-based distance)
ks = KShape(n_clusters=k, verbose=True, random_state=42)
y_ks = ks.fit_predict(X_scaled)

# Calculate Adjusted Rand Index for KShape
ari_ks = adjusted_rand_score(y, y_ks)
print(f"KShape - Adjusted Rand Index: {ari_ks:.4f}")

# Plot KShape clustering results
plot_ks = plot_clustering_results(X_scaled, y_ks, "KShape Clustering Results")
plot_ks.savefig('graphs/kshape_clusters.png')

# ==========================================
# Method 4: DBSCAN with DTW
# ==========================================
print("Performing DBSCAN clustering with DTW...")

# We need to determine a good value for epsilon (distance threshold)
# Let's use the knee/elbow method by looking at the k-distance graph

# Sort pairwise distances in ascending order for each point
distances = np.sort(dtw_matrix, axis=1)
# Get the distances to the kth nearest neighbor
k_distances = distances[:, 1:6]  # Get distances to the 1st through 5th nearest neighbors
# Calculate average k-distance
avg_k_distances = np.mean(k_distances, axis=1)
# Sort in ascending order for the k-distance graph
sorted_k_distances = np.sort(avg_k_distances)

# Estimate a good epsilon value based on the knee/elbow of the k-distance graph
# Using a simple heuristic - we can improve this if needed

# Find the "knee" in the k-distance curve
kneedle = KneeLocator(
    range(len(sorted_k_distances)),
    sorted_k_distances,
    S=1.0,
    curve="convex",
    direction="increasing"
)

if kneedle.knee is not None:
    epsilon = sorted_k_distances[kneedle.knee]
else:
    # If knee point can't be determined automatically, set a reasonable value
    # This is based on the average distance to approximately k/2 nearest neighbors
    epsilon = np.median(sorted_k_distances)

print(f"Selected epsilon for DBSCAN: {epsilon:.4f}")

# Mark the selected epsilon in the k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(sorted_k_distances)
plt.axhline(y=epsilon, color='r', linestyle='--', label=f'Epsilon = {epsilon:.4f}')
plt.title('K-distance Graph (k=5) with Selected Epsilon')
plt.xlabel('Points sorted by distance')
plt.ylabel('Average distance to k nearest neighbors')
plt.legend()
plt.grid(True)
plt.savefig('graphs/dbscan_epsilon_selection.png')

# Apply DBSCAN with the selected epsilon
# Min_samples is set to be roughly log(n_samples) which is a common heuristic
min_samples = max(int(np.log(n_samples)), 2)
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed')
y_dbscan = dbscan.fit_predict(dtw_matrix)

# Handle noise points (-1 label) for visualization purposes
if -1 in y_dbscan:
    # Count the number of clusters excluding noise
    n_clusters_dbscan = len(np.unique(y_dbscan[y_dbscan >= 0]))
    print(f"DBSCAN detected {n_clusters_dbscan} clusters and {np.sum(y_dbscan == -1)} noise points")

    # Create a special plot for DBSCAN that includes noise points
    plt.figure(figsize=(12, 8))

    # First plot the noise points
    plt.subplot(n_clusters_dbscan + 1, 1, 1)
    for ts in X_scaled[y_dbscan == -1]:
        plt.plot(ts.ravel(), 'k-', alpha=0.2)
    plt.title('Noise Points')

    # Then plot each cluster
    for i, cls in enumerate(np.unique(y_dbscan[y_dbscan >= 0])):
        plt.subplot(n_clusters_dbscan + 1, 1, i + 2)
        for ts in X_scaled[y_dbscan == cls]:
            plt.plot(ts.ravel(), 'k-', alpha=0.2)
        mean_ts = np.mean(X_scaled[y_dbscan == cls], axis=0)
        plt.plot(mean_ts.ravel(), 'r-', linewidth=2)
        plt.title(f'Cluster {cls}')

    plt.suptitle('DBSCAN Clustering Results with DTW', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('graphs/dbscan_clusters.png')
else:
    # If there are no noise points, use the standard plotting function
    plot_dbscan = plot_clustering_results(X_scaled, y_dbscan, "DBSCAN Clustering Results with DTW")
    plot_dbscan.savefig('graphs/dbscan_clusters.png')

# Calculate ARI for DBSCAN (if there are at least 2 clusters)
if len(np.unique(y_dbscan)) >= 2:
    ari_dbscan = adjusted_rand_score(y, y_dbscan)
    print(f"DBSCAN - Adjusted Rand Index: {ari_dbscan:.4f}")
else:
    print("DBSCAN did not find meaningful clusters with the current parameters")
    ari_dbscan = 0.0

# ==========================================
# Comparison of Methods
# ==========================================
# Create a comparison table
methods = ['DTW K-means', 'Hierarchical Clustering', 'KShape', 'DBSCAN']
ari_scores = [ari_km_dtw, ari_hc, ari_ks, ari_dbscan]
silhouette_scores = [sil_km_dtw, sil_hc, np.nan, np.nan]  # KShape and DBSCAN don't use the same distance metric

# Create a DataFrame for the comparison
comparison_df = pd.DataFrame({
    'Method': methods,
    'Adjusted Rand Index': ari_scores,
    'Silhouette Score': silhouette_scores
})

print("\nComparison of Methods:")
print(comparison_df)

# Plot the comparison of ARI scores
plt.figure(figsize=(10, 6))
plt.bar(methods, ari_scores, color=['blue', 'green', 'orange', 'purple'])
plt.title('Comparison of Clustering Methods (Adjusted Rand Index)')
plt.ylabel('Adjusted Rand Index')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('graphs/method_comparison.png')


# Create confusion matrices for each method
plot_confusion_matrix(y, y_km_dtw, 'Confusion Matrix: DTW K-means').savefig('graphs/confusion_kmeans_dtw.png')
plot_confusion_matrix(y, y_hc, 'Confusion Matrix: Hierarchical Clustering').savefig('graphs/confusion_hierarchical.png')
plot_confusion_matrix(y, y_ks, 'Confusion Matrix: KShape').savefig('graphs/confusion_kshape.png')

# Create confusion matrix for DBSCAN only if it found more than one cluster
if len(np.unique(y_dbscan)) >= 2:
    plot_confusion_matrix(y, y_dbscan, 'Confusion Matrix: DBSCAN').savefig('graphs/confusion_dbscan.png')
