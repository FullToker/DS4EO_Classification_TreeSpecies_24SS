import numpy as np
from loader import File_geojson
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt

X = np.load("./model/data/X_augm.npy")
y = np.load("./model/data/y_augm.npy")

X_test = np.load("./test/my_patches.npy")
y_test = np.load("./test/my_labels.npy")

for i in range(10):
    count = 0
    for label in y_test:
        if label == i:
            count += 1
    print(f"the label {i + 1} : {count}")

# pca_test = KernelPCA(n_jobs=-1, n_components=128, kernel="rbf")
# # pca_test = PCA(n_components=128)
# pca_test.fit(X)
# X_pca = pca_test.transform(X)

# plt.figure(figsize=(10, 7))
# plt.subplot(1, 2, 1)
# scatter = plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1],
#     label=y[:],
#     c=y,
#     cmap=plt.cm.get_cmap("tab10", 10),
#     alpha=0.5,
# )
# plt.colorbar(scatter, label="Label")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.subplot(1, 2, 2)
# scatter = plt.scatter(
#     X_pca[:, 2],
#     X_pca[:, 3],
#     label=y[:],
#     c=y,
#     cmap=plt.cm.get_cmap("tab10", 10),
#     alpha=0.5,
# )
# plt.colorbar(scatter, label="Label")
# plt.xlabel("Principal Component 3")
# plt.ylabel("Principal Component 4")
# plt.show()

# # for Test set -----------------
# pca_test.fit(X_test)
# X_pca_test = pca_test.transform(X_test)

# plt.figure(figsize=(10, 7))
# plt.subplot(1, 2, 1)
# scatter = plt.scatter(
#     X_pca_test[:, 0],
#     X_pca_test[:, 1],
#     label=y_test[:],
#     c=y_test,
#     cmap=plt.cm.get_cmap("tab10", 10),
#     alpha=0.5,
# )
# plt.colorbar(scatter, label="Label")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.subplot(1, 2, 2)
# scatter = plt.scatter(
#     X_pca_test[:, 2],
#     X_pca_test[:, 3],
#     label=y_test[:],
#     c=y_test,
#     cmap=plt.cm.get_cmap("tab10", 10),
#     alpha=0.5,
# )
# plt.colorbar(scatter, label="Label")
# plt.xlabel("Principal Component 3")
# plt.ylabel("Principal Component 4")
# plt.show()

#
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

print("it's the test time")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RF(n_estimators=100)
model.fit(X_train, y_train)

# val
y_pred = model.predict(X_val)

cm = confusion_matrix(y_val, y_pred)
print(f"accuracy of val: {np.sum(y_pred == y_val) / len(y_pred)}")

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
# plt.xlabel("Predicted label")
# plt.ylabel("Ground Truth")
# plt.title("Confusion Matrix for RF")
# plt.show()


# test

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(f"accuracy of test: {np.sum(y_pred == y_test) / len(y_pred)}")

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
# plt.xlabel("Predicted label")
# plt.ylabel("Ground Truth")
# plt.title("Confusion Matrix for RF")
# plt.show()


print("Done!")
