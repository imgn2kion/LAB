import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "Feature 1": [2.5, 0.5, 2.2, 1.9, 3.1],
    "Feature 2": [2.4, 0.7, 2.9, 2.2, 3.0],
    "Feature 3": [3.6, 0.9, 3.1, 2.7, 3.8]
}

df = pd.DataFrame(data)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(standardized_data)
pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
evr = pca.explained_variance_ratio_
print("Explained Variance Ratio:", evr)
eigenvectors = pca.components_
print("Principal Components (Eigenvectors):\n", eigenvectors)
print("\nTransformed Dataset:")
print(pca_df)
print("\nOutput DataFrame:")
print(pca_df)
