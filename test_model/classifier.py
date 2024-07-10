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
import matplotlib.pyplot as plt

X = np.load("./test_model/data/X_norm.npy")
y = np.load("./test_model/data/y_norm.npy")

# X = np.load("./test_model/data/X_norm_augm.npy")
# y = np.load("./test_model/data/y_norm_augm.npy")

print(f"data shape: {X.shape}")
print(f"label shape: {y.shape}")

X_test = np.load("./test_model/data/X_eval.npy")
y_test = np.load("./test_model/data/y_eval.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")

# pca to reduce dimensions
pca = PCA(n_components=25)
# pca = KernelPCA(n_jobs=-1, n_components=25, kernel="rbf", gamma=5)

# reduce dimensions
pca.fit(X_train)
X_train_reduced = pca.transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)
print(f"shape of x_test_reduced: {X_test_reduced.shape}")


classifiers = {
    # "1KNN": KNeighborsClassifier(n_neighbors=1),
    # "3KNN": KNeighborsClassifier(n_neighbors=3),
    # "5KNN": KNeighborsClassifier(n_neighbors=5),
    "DT": tree.DecisionTreeClassifier(),
    "NB": GaussianNB(),
    "RF": RDF(n_estimators=100),
    "MLP": MLPClassifier(alpha=1, max_iter=1000, activation="relu"),
    # "SVM": SVC(),
    "newSVC": SVC(gamma=2, C=1),
}

y__ = dict()
accuracy = dict()
print("accuracy of each classifiers in Val set is:")
for clf_name, clf in classifiers.items():
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_val_reduced)
    y__[clf_name] = y_pred
    acc = np.sum(y__[clf_name] == y_val) / len(y_pred)
    print(f"{clf_name: >15}: {100*acc:.2f}%")

# in test set
y__ = dict()
accuracy = dict()
print("accuracy of each classifiers in Test set is:")
for clf_name, clf in classifiers.items():
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_test_reduced)
    y__[clf_name] = y_pred
    acc = np.sum(y__[clf_name] == y_test) / len(y_pred)
    print(f"{clf_name: >15}: {100*acc:.2f}%")
