from math import gamma
from loader import Feature, File_geojson
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    print(f"{treeclass.get_name()} : {treeclass.get_feature_num()}")


#
X = np.concatenate(imgs, axis=0)
y = np.concatenate(labels, axis=0)
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")


img_size = 25
bands_num = len(X[0]) // img_size
pca = PCA(n_components=3)
pca_bands = []
for i in range(bands_num):
    sub_band = X[:, i * 25 : (i + 1) * 25]
    # print(sub_band.shape)
    band_pca = pca.fit_transform(sub_band)
    pca_bands.append(band_pca)
pca_bands = np.stack(pca_bands, axis=1)
pca_X = pca_bands.reshape((len(pca_bands), -1))
print(pca_X.shape)

# Feature Selection ------------------------------------------//////////
select_func = SVC(gamma=2, C=1)
# select_func = RF(n_estimators=100)
# SFFS
sfs = SFS(select_func, n_features_to_select=15, cv=5, n_jobs=-1)
sfs.fit(pca_X, y)
print(sfs.get_support())
np.save("features_sfs_svm15.npy", sfs.get_support())
new_X = sfs.transform(pca_X)

# RFECV
"""
rfe = RFECV(select_func, n_jobs=-1, cv=5, min_features_to_select=15)
rfe.fit(pca_X, y)
print(rfe.get_support())
new_X = rfe.transform(pca_X)
"""


X_train, X_val, y_train, y_val = train_test_split(
    new_X, y, test_size=0.2, random_state=42
)
print(f"X_train Shape: {X_train.shape}")
print(f"y_train : {y_train.shape}")


plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
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
plt.subplot(1, 2, 2)
scatter = plt.scatter(
    X_train[:, 2],
    X_train[:, 3],
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
    # "3KNN": KNeighborsClassifier(n_neighbors=3),
    "5KNN": KNeighborsClassifier(n_neighbors=5),
    "NB": GaussianNB(),
    "RF": RF(n_estimators=100),
    "MLP": MLPClassifier(alpha=1, max_iter=1000, activation="relu"),
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

# plot the confusion matrix of the SVM Classifier
plot_model = SVC(gamma=2, C=1)
plot_model.fit(X_train, y_train)
y_pred = plot_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.xlabel("Predicted label")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix for SVM")
plt.show()


print("done")
