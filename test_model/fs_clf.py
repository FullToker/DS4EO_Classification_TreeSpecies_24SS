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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import umap
from sklearn.feature_selection import SequentialFeatureSelector as SFS

X = np.load("./test_model/data/X_norm.npy")
y = np.load("./test_model/data/y_norm.npy")

# X = np.load("./test_model/data/X_norm_augm.npy")
# y = np.load("./test_model/data/y_norm_augm.npy")

print(f"data shape: {X.shape}")
print(f"label shape: {y.shape}")

X_test = np.load("./test_model/data/X_eval_new.npy")
y_test = np.load("./test_model/data/y_eval_new.npy")

"""
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")
"""


supports = np.load("./test_model/sfs_rf90.npy").reshape(-1)
print(supports.shape)
# new_X = sfs.transform(X)
new_X = X[:, supports]
X_test = X_test[:, supports]

X_train, X_val, y_train, y_val = train_test_split(
    new_X, y, test_size=0.2, random_state=42
)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")

# classification model

classifiers = {
    # "1KNN": KNeighborsClassifier(n_neighbors=1),
    # "3KNN": KNeighborsClassifier(n_neighbors=3),
    "5KNN": KNeighborsClassifier(n_neighbors=5),
    "DT": tree.DecisionTreeClassifier(),
    "NB": GaussianNB(),
    "RF": RDF(n_estimators=100),
    "MLP": MLPClassifier(alpha=1, max_iter=2000, activation="relu"),
    # "SVM": SVC(),
    "newSVC": SVC(gamma=2, C=1),
}

y__ = dict()
accuracy = dict()
print("accuracy of each classifiers is:")
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y__[clf_name] = y_pred
    acc = np.sum(y__[clf_name] == y_val) / len(y_pred)
    print(f"{clf_name: >15}: {100*acc:.2f}%")


# in test set
y__ = dict()
accuracy = dict()
print("accuracy of each classifiers in Test set is:")
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y__[clf_name] = y_pred
    acc = np.sum(y__[clf_name] == y_test) / len(y_pred)
    print(f"{clf_name: >15}: {100*acc:.2f}%")

    """   
    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {clf_name}")
    plt.show()
    """
