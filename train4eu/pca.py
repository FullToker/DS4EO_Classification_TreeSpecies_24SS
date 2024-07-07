import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load the dataset: delete samples with null; without augmentation
X = np.load("../dataset/X_all_outnull.npy")
y = np.load("../dataset/y_all_outnull.npy")

# Split the train and test dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")

# already use the PCA -> use the TSNE
tsne = SpectralEmbedding(n_jobs=-1, affinity="rbf", gamma=2, n_components=2)
tsne.fit(X_train)
X_train = tsne.fit_transform(X_train)

# show the 4
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    label=y_train[:],
    c=y_train,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
