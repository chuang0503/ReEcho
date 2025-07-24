import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class TSNE_KMeans:
    def __init__(self, perplexity=50, n_iter_without_progress=1000, n_clusters=10, random_state=42):
        self.perplexity = perplexity
        self.n_iter_without_progress = n_iter_without_progress
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.tsne = TSNE(n_components=2, perplexity=perplexity, n_iter_without_progress=n_iter_without_progress, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    
    def fit(self, rir_feat_list):
        X = self.scaler.fit_transform(rir_feat_list)
        X_embedded = self.tsne.fit_transform(X)
        self.kmeans.fit(X_embedded)
        return self.kmeans.labels_
