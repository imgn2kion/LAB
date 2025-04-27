import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

k = 3

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

labels = kmeans.labels_

print("Point ID\tx1\tx2\tCluster")
for i in range(len(X)):
    print(f"{i+1}\t{X[i][0]}\t{X[i][1]}\t{labels[i]}")

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'KMeans Clustering (k={k})')
plt.colorbar(label='Cluster')
plt.show()
