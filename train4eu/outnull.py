from loader import File_geojson
import numpy as np


# read the geojson file ---------------------------////////////////
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
data_path = [
    ("../dataset/all_trees/" + x + "_0528.geojson") for _, x in label_dict.items()
]

# all classes
imgs = []
labels = []

for i in range(len(label_dict)):
    treeclass = File_geojson(data_path[i], label_dict)
    imgs.append(treeclass.get_finaldata())
    labels.append(treeclass.get_labels())
    name = treeclass.get_name()
    print(f" {name}: {treeclass.get_real_feature_num()}")

#
X = np.concatenate(imgs, axis=0)
y = np.concatenate(labels, axis=0)
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")
np.save("../dataset/X_all_outnull.npy", X)
np.save("../dataset/y_all_outnull.npy", y)
