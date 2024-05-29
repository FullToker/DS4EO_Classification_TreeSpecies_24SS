from io import FileIO
import numpy as np
import re
import json
from geojson import FeatureCollection

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
    ("./Dataset/norepeat/" + x + "_0517.geojson") for _, x in label_dict.items()
]


class File_geojson:

    def __init__(self, path, label_dict):
        self.ord_path = path
        # load json file
        with open(path, "r") as f:
            self.all_data = json.load(f)
        self.feature_collection = FeatureCollection(self.all_data)
        self.num = len(self.feature_collection["features"])
        # load all features
        self.all_features = []
        for i in range(len(self.feature_collection["features"])):
            self.all_features.append(
                self.feature_collection["features"][i]["properties"]
            )

        # label
        pattern = r"(?<=\/)([^\/]+)(?=_0517)"
        self.name = re.search(pattern, path).group(0)
        label = [key for key, value in label_dict.items() if value == self.name][0]
        self.labels = np.array([label] * self.num)

    def process_data(self, feature):
        # feature is a dict consists of 3 * 10 bands: B2 - B12_2;
        # each band is (5,5) all 25pixels
        self.bands = [
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
            "B2_1",
            "B3_1",
            "B4_1",
            "B5_1",
            "B6_1",
            "B7_1",
            "B8_1",
            "B8A_1",
            "B11_1",
            "B12_1",
            "B2_2",
            "B3_2",
            "B4_2",
            "B5_2",
            "B6_2",
            "B7_2",
            "B8_2",
            "B8A_2",
            "B11_2",
            "B12_2",
        ]
        flatten_data = []  #  length =30
        for band in self.bands:
            img = feature[band]
            # deal with the "null" and "nonetype"(caused by no this band)
            if type(img) == type(None):
                img = np.zeros((5, 5))
            flatten_data.append(np.array(img).reshape(-1))
        all_imgs = np.concatenate(flatten_data)
        return all_imgs

    def get_finaldata(self):
        feature_imgs = []
        for i in range(self.num):
            feature = self.all_features[i]
            each_imgs = self.process_data(feature)
            if len(each_imgs) != len(self.bands) * 25:
                print(f"第{i}个feature的大小不对")
            feature_imgs.append(each_imgs)

        # get all flatten imgs in one class array(num, 750)
        self.newdata = np.stack(feature_imgs)
        assert self.newdata.shape == (
            self.num,
            len(self.bands) * 25,
        ), "the shape of final data is not correct, != (num, 750)"

        return self.newdata

    # get the name of this class
    def get_name(self):
        return self.name

    # get specific feature accoding to index
    def get_feature_index(self, index: int):
        feature = self.all_features[index]
        return feature

    # get the all number of features
    def get_fearture_num(self):
        return self.num

    # get the labels(np.array)
    def get_labels(self):
        return self.labels


for i in range(10):
    treeclass = File_geojson(data_path[i], label_dict)
    print(treeclass.get_name())
    print(treeclass.get_fearture_num())
    treeclass.get_finaldata()

# visualization of each class with 30 bands
# show the first img of picea
import matplotlib.pyplot as plt

# show all classes' first image
for path in data_path:
    treeclass = File_geojson(path, label_dict)
    final = treeclass.get_finaldata()
    img = final[0].reshape(30, 5, 5)
    img *= 255

    fig, axes = plt.subplots(3, 10, figsize=(5, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img[i], extent=[0, 2, 0, 2])
        ax.axis("off")
    plt.show()

"""
# test for feature
acer = File_geojson(data_path[7], label_dict)
final = acer.get_finaldata()
label = acer.get_labels()
print(f"final's shape is {final.shape}")
print(f"labels' shape is {label.shape}")
"""
