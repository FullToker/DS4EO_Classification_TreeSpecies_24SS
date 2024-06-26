import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.feature_selection import SequentialFeatureSelector as SFS

X = np.load("./data/ksh/nodup_patches.npy")
# X = np.load("./model/data/X_augm.npy")
X = X.reshape(len(X), -1)
y = np.load("./data/ksh/nodup_labels.npy")
# y = np.load("./model/data/y_augm.npy")

print(X.shape)
print(y.shape)

X_test = np.load("./test/test_patches.npy")
X_test = X_test.reshape(len(X_test), -1)
y_test = np.load("./test/test_labels.npy")


img_size = 25
bands_num = len(X[0]) // img_size
pca = KernelPCA(n_components=5, n_jobs=-1, kernel="rbf")
pca_bands = []
pca_test_bands = []
for i in range(bands_num):
    sub_band = X[:, i * 25 : (i + 1) * 25]
    # print(sub_band.shape)
    pca.fit(sub_band)
    band_pca = pca.transform(sub_band)
    pca_bands.append(band_pca)

    # for the test data
    test = pca.transform(X_test[:, i * 25 : (i + 1) * 25])
    pca_test_bands.append(test)

pca_bands = np.stack(pca_bands, axis=1)
pca_X = pca_bands.reshape((len(pca_bands), -1))
print(pca_X.shape)

pca_test = np.stack(pca_test_bands, axis=-1)  # for test data's sffs
pca_test = pca_test.reshape((len(pca_test), -1))
print(f"pca_test's shape: {pca_test.shape}")

"""
# Feature Selection ------------------------------------------//////////
# support = np.load("./model/feature_selection/features_sfs_svm15.npy")
select_func = RF(n_estimators=100, n_jobs = -1)
# SFFS
sfs = SFS(select_func, n_features_to_select=30, cv=5, n_jobs=-1)
sfs.fit(pca_X, y)
print(sfs.get_support())
np.save("./model/feature_selection/features_sfs_rf25_olddata.npy", sfs.get_support())
support = sfs.get_support()
"""

support = np.load("./model/feature_selection/features_sfs_rf25_olddata.npy")


X = pca_X[:, support]
X_test = pca_test[:, support]

# just use 90 dims
X = pca_X
X_test = pca_test

# ///////////////////--------------------------------/////////////////////////
# plot the pca
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
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
    X[:, 2],
    X[:, 3],
    label=y[:],
    c=y,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 3")
plt.ylabel("Principal Component 4")
plt.show()

# do the feature selection for the test set
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
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
    X_test[:, 2],
    X_test[:, 3],
    label=y_test[:],
    c=y_test,
    cmap=plt.cm.get_cmap("tab10", 10),
    alpha=0.5,
)
plt.colorbar(scatter, label="Label")
plt.xlabel("Principal Component 3")
plt.ylabel("Principal Component 4")
plt.show()


# -----------------------------------//////////////////////-------------------
print("beagin train!")
model = RF(n_estimators=100, n_jobs=-1)
model.fit(X, y)


y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"score: {model.score(X_test, y_test)}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.xlabel("Predicted label")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix for RF")
plt.show()
