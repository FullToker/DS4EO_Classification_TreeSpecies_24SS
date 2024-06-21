import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

X = np.load("./data/ksh/nodup_patches.npy")
X = np.load("./model/data/X_augm.npy")
X = X.reshape(len(X), -1)
y = np.load("./data/ksh/nodup_labels.npy")
y = np.load("./model/data/y_augm.npy")

print(X.shape)
print(y.shape)


from sklearn.decomposition import PCA

pca = PCA(n_components=25)

pca.fit(X)
X = pca.transform(X)
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


from sklearn.ensemble import RandomForestClassifier as RF

model = RF(n_estimators=100)
model.fit(X, y)

X_test = np.load("./test/my_patches.npy")
X_test = X_test.reshape(len(X_test), -1)
y_test = np.load("./test/my_labels.npy")

X_test = pca.transform(X_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"score: {model.score(X_test, y_test)}")
print(f"accuracy of val: {np.sum(y_pred == y_test) / len(y_pred)}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.xlabel("Predicted label")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix for RF")
plt.show()
