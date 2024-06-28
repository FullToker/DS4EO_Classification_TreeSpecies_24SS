import numpy as np
from sklearn.base import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from autoencoder import AutoEncoder, Encoder, Decoder
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

encoder = Encoder()
encoder.load_state_dict(torch.load("./encoder/save/encoder.pth"))

# load the dataset
train_data = torch.load("./encoder/save/train.pth")
test_data = torch.load("./encoder/save/test.pth")

train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

X = []
y = []
with torch.no_grad():
    for data, label in train_loader:
        encoded = encoder.forward(data)
        X.append(encoded.view(encoded.size(0), -1).numpy())
        y.append(label.numpy())

X = np.vstack(X)
y = np.hstack(y)
print(X.shape)

X_val = []
y_val = []
with torch.no_grad():
    for data, label in test_loader:
        encoded = encoder.forward(data)
        X_val.append(encoded.view(encoded.size(0), -1).numpy())
        y_val.append(label.numpy())

X_val = np.vstack(X_val)
y_val = np.hstack(y_val)


classifiers = {
    # "1KNN": KNeighborsClassifier(n_neighbors=1),
    # "3KNN": KNeighborsClassifier(n_neighbors=3),
    "5KNN": KNeighborsClassifier(n_neighbors=5),
    "DT": tree.DecisionTreeClassifier(),
    "NB": GaussianNB(),
    "RF": RDF(n_estimators=100),
    "MLP": MLPClassifier(alpha=1, max_iter=1000, activation="relu"),
    # "SVM": SVC(),
    "newSVC": SVC(gamma=2, C=1),
}

print("accuracy of each classifiers is:")
for clf_name, clf in classifiers.items():
    clf.fit(X, y)
    y_pred = clf.predict(X_val)
    acc = clf.score(y_pred, y_val)
    print(f"{clf_name: >15}: {100*acc:.2f}%")