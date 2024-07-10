import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.ensemble import RandomForestClassifier as RDF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# X = np.load("./test_model/data/X_norm.npy")
# y = np.load("./test_model/data/y_norm.npy")

X = np.load("./test_model/data/X_norm_augm.npy")
y = np.load("./test_model/data/y_norm_augm.npy")

print(f"data shape: {X.shape}")
print(f"label shape: {y.shape}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")

# pca to reduce dimensions
pca = PCA(n_components=20)
X_train_reduced = pca.fit_transform(X_train)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_train_reduced[:, 0],
    X_train_reduced[:, 1],
    label=y_train[:],
    c=y_train,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.subplot(1, 2, 2)
scatter = plt.scatter(
    X_train_reduced[:, 2],
    X_train_reduced[:, 3],
    label=y_train[:],
    c=y_train,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 3")
plt.ylabel("Principal Component 4")
plt.show()
