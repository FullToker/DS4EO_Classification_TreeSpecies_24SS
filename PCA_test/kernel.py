import numpy as np
from loader import File_geojson
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt

X = np.load("./model/data/X_augm.npy")
y = np.load("./model/data/y_augm.npy")

X_test = np.load("./test/my_patches.npy")
y_test = np.load("./test/my_labels.npy")


for i 

# pca_test = KernelPCA(n_jobs=-1, n_components=15, kernel="rbf")
pca_test = PCA(n_components=15)
pca_test.fit(X)
X_pca = pca_test.transform(X)

plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    label=y[:],
    c=y,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.subplot(1, 2, 2)
scatter = plt.scatter(
    X_pca[:, 2],
    X_pca[:, 3],
    label=y[:],
    c=y,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 3")
plt.ylabel("Principal Component 4")
plt.show()

# for Test set -----------------
pca_test.fit(X_test)
X_pca = pca_test.transform(X_test)

plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    label=y_test[:],
    c=y_test,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.subplot(1, 2, 2)
scatter = plt.scatter(
    X_pca[:, 2],
    X_pca[:, 3],
    label=y_test[:],
    c=y_test,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 3")
plt.ylabel("Principal Component 4")
plt.show()
