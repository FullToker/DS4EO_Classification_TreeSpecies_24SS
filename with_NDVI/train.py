import numpy as np
from loader import File_geojson
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RDF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

label_dict = {
    0: "Picea abies",
    1: "Fagus sylvatica",
    2: "Pinus sylvestris",
    3: "Quercus robur",
    4: "Betula pendula",
    5: "Quercus petraea",
    6: "Fraxinus excelsior",
    7: "Acer pseudoplatanus",
    8: "Sorbus aucuparia",
    9: "Carpinus betulus",
}
data_path = [("./data/test/" + x + "_0528.geojson") for _, x in label_dict.items()]

# all classes
imgs = []
labels = []

for i in range(len(label_dict)):
    treeclass = File_geojson(data_path[i], label_dict)
    imgs.append(treeclass.get_finaldata())
    labels.append(treeclass.get_labels())
    name = treeclass.get_name()
    print(f" {name}: {treeclass.get_feature_num()}")

#
X = np.concatenate(imgs, axis=0)
y = np.concatenate(labels, axis=0)
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")
np.save("./data/np_data/X_ndvi.npy", X)
np.save("./data/np_data/y_ndvi.npy", y)


# counts = np.bincount(y)
# for i, count in enumerate(counts):
#    print(f"label {i}: {count}")

# Using PCA to dimension reduction ----------------////////////
# begin model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")

# pca to reduce dimensions
pca = PCA(n_components=15)
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


classifiers = {
    # "1KNN": KNeighborsClassifier(n_neighbors=1),
    "3KNN": KNeighborsClassifier(n_neighbors=3),
    "5KNN": KNeighborsClassifier(n_neighbors=5),
    "DT": tree.DecisionTreeClassifier(),
    "NB": GaussianNB(),
    "RF": RDF(n_estimators=100),
    "MLP": MLPClassifier(alpha=1, max_iter=1000, activation="relu"),
    # "SVM": SVC(),
    "newSVC": SVC(gamma=2, C=1),
}
X_val_reduced = pca.fit_transform(X_val)

y__ = dict()
accuracy = dict()
print("accuracy of each classifiers is:")
for clf_name, clf in classifiers.items():
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_val_reduced)
    y__[clf_name] = y_pred
    acc = np.sum(y__[clf_name] == y_val) / len(y_pred)
    print(f"{clf_name: >15}: {100*acc:.2f}%")