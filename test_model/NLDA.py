import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.ensemble import RandomForestClassifier as RDF
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import umap
import umap.plot

X = np.load("./test_model/data/X_norm.npy")
y = np.load("./test_model/data/y_norm.npy")

# X = np.load("./test_model/data/X_norm_augm.npy")
# y = np.load("./test_model/data/y_norm_augm.npy")

print(f"data shape: {X.shape}")
print(f"label shape: {y.shape}")

X_test = np.load("./test_model/data/X_eval_new.npy")
y_test = np.load("./test_model/data/y_eval_new.npy")


# split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")

# pca to reduce dimensions
# pca = PCA(n_components=25)
# pca = KernelPCA(n_jobs=-1, n_components=25, kernel="rbf", gamma=1)
# pca = TruncatedSVD(n_components=25)
pca = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
# pca = SpectralEmbedding(n_components=2)

# reduce dimensions
mapper = pca.fit(X_train)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)
print(f"shape of x_test_reduced: {X_test_reduced.shape}")
umap.plot.points(mapper)
umap.plot.points(X_test_reduced)

plt.figure(figsize=(10, 7))
# plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_train_reduced[:, 0],
    X_train_reduced[:, 1],
    label=y_train[:],
    c=y_train,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Embedded Space 1")
plt.ylabel("Embedded Space 2")
# plt.savefig("./test_model/runs/TSNE_norm_3.png")
plt.show()

"""
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
plt.xlabel("singular vectors 3")
plt.ylabel("singular vectors 4")
plt.show()

# draw the testset
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_test_reduced[:, 0],
    X_test_reduced[:, 1],
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
    X_test_reduced[:, 2],
    X_test_reduced[:, 3],
    label=y_test[:],
    c=y_test,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 3")
plt.ylabel("Principal Component 4")
plt.show()
"""
