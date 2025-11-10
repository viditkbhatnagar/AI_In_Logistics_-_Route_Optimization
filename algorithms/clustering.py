"""
Clustering algorithms for delivery zone optimization
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import CLUSTERING_MIN_K, CLUSTERING_MAX_K


class DeliveryZoneOptimizer:
    """K-means clustering for delivery zone optimization"""
    
    def __init__(self, delivery_locations, min_k=CLUSTERING_MIN_K, max_k=CLUSTERING_MAX_K):
        self.delivery_locations = delivery_locations
        self.min_k = min_k
        self.max_k = max_k
        self.coordinates = delivery_locations[['latitude', 'longitude']].values
        self.best_k = None
        self.best_model = None
        self.cluster_labels = None
        self.silhouette_scores = []
    
    def find_optimal_k(self):
        """Find optimal number of clusters using elbow method and silhouette score"""
        k_range = range(self.min_k, min(self.max_k + 1, len(self.delivery_locations)))
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.coordinates)
            
            inertias.append(kmeans.inertia_)
            silhouette = silhouette_score(self.coordinates, labels)
            silhouette_scores.append(silhouette)
        
        self.silhouette_scores = silhouette_scores
        
        # Choose k with highest silhouette score
        best_idx = np.argmax(silhouette_scores)
        self.best_k = list(k_range)[best_idx]
        
        return list(k_range), inertias, silhouette_scores
    
    def fit(self, n_clusters=None):
        """Fit clustering model"""
        if n_clusters is None:
            # Find optimal k
            k_range, inertias, silhouette_scores = self.find_optimal_k()
            n_clusters = self.best_k
        else:
            self.best_k = n_clusters
        
        # Fit model
        self.best_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.best_model.fit_predict(self.coordinates)
        
        return self.cluster_labels
    
    def get_cluster_centers(self):
        """Get cluster centers"""
        if self.best_model is None:
            return None
        return self.best_model.cluster_centers_
    
    def get_cluster_stats(self):
        """Get statistics for each cluster"""
        if self.cluster_labels is None:
            return None
        
        stats = []
        for cluster_id in range(self.best_k):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_locations = self.delivery_locations[cluster_mask]
            
            center = self.best_model.cluster_centers_[cluster_id]
            
            stats.append({
                'cluster_id': cluster_id,
                'n_deliveries': len(cluster_locations),
                'center_latitude': center[0],
                'center_longitude': center[1],
                'avg_latitude': cluster_locations['latitude'].mean(),
                'avg_longitude': cluster_locations['longitude'].mean()
            })
        
        return pd.DataFrame(stats)
    
    def assign_to_clusters(self, new_locations):
        """Assign new locations to clusters"""
        if self.best_model is None:
            raise ValueError("Model must be fitted first")
        
        new_coords = new_locations[['latitude', 'longitude']].values
        labels = self.best_model.predict(new_coords)
        return labels

