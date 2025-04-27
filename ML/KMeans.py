import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
])

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
labels_kmeans = kmeans.labels_

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)
labels_gmm = gmm.predict(X)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x')
plt.title("K-Means Clustering")
plt.xlabel('x1')
plt.ylabel('x2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis', marker='o')
plt.title("EM (Gaussian Mixture) Clustering")
plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.show()
