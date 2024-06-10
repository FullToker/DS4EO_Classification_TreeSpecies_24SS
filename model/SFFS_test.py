from loader import Feature, File_geojson
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier as RF

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
    # print(treeclass.get_feature_num())


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


# SFFS
rf = RF(n_estimators=100)
sfs = SFS(rf, n_features_to_select=3, cv=5, n_jobs=-1)
sfs.fit(pca_X, y)
print(sfs.get_support())
