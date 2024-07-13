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

X_test = np.load("./test_model/data/X_eval_new.npy")
y_test = np.load("./test_model/data/y_eval_new.npy")


# split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")

"""
# pca to reduce dimensions
pca = PCA(n_components=25)
# pca = KernelPCA(n_jobs=-1, n_components=25, kernel="rbf", gamma=1)
# pca = TruncatedSVD(n_components=25)

# reduce dimensions
pca.fit(X_train)
X_train_reduced = pca.transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)
print(f"shape of x_test_reduced: {X_test_reduced.shape}")
"""

# begin the test
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

# using dict to save the result of the each classifier
acc_dicts = {"DT": [], "NB": [], "RF": [], "MLP": [], "newSVC": []}
numbers = [5 * i for i in range(1, 41)]

for i in numbers:
    pca = PCA(n_components=i)
    pca.fit(X_train)
    X_train_reduced = pca.transform(X_train)
    X_val_reduced = pca.transform(X_val)
    X_test_reduced = pca.transform(X_test)
    print(f"shape of x_test_reduced: {X_test_reduced.shape}")

    # using the val set to see if the number of components affects the performance
    y__ = dict()
    accuracy = dict()
    print(f"accuracy of each classifiers with {i} Components in Val set is:")
    for clf_name, clf in classifiers.items():
        clf.fit(X_train_reduced, y_train)
        y_pred = clf.predict(X_val_reduced)
        y__[clf_name] = y_pred
        acc = np.sum(y__[clf_name] == y_val) / len(y_pred)
        acc_dicts[clf_name].append(acc)
        # print(f"{clf_name: >15}: {100*acc:.2f}%")

# plot the comparison
plt.figure(figsize=(10, 6))
for clf_name, accuracies in acc_dicts.items():
    plt.plot(numbers, accuracies, label=clf_name)

plt.xlabel("Number of Components")
plt.ylabel("Accuracy")
plt.title("Accuracy of Different Classifiers")
plt.legend()
plt.grid(True)
# plt.savefig("accuracy_plot.png")
plt.show()

"""
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

    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {clf_name}")
    plt.savefig(f"./visual/CM_{clf_name}_TSNE_norm_50.png")
    # plt.show()
"""