from io import FileIO
import numpy as np
import re
import json
from geojson import FeatureCollection
from sklearn.decomposition import PCA


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
        pattern = r"(?<=\/)([^\/]+)(?=_0528)"
        self.name = re.search(pattern, path).group(0)
        label = [key for key, value in label_dict.items() if value == self.name][0]
        self.labels = np.array([label] * self.num)

        self.num_newband = 0
        # record the delete the index
        self.delete_index = []

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
                # continue

            if len(img) != 5:
                # print(img)
                img = np.zeros((5, 5))
            flatten_data.append(np.array(img).reshape(-1))

        """         
        # we want to add NDVI band for each season
        season = int(len(self.bands) / 10)
        assert season == 3, "the number of bands cannot devide by 10"
        self.num_newband = 3
        for i in range(season):
            img = np.zeros((5, 5))
            img = np.array(img).reshape(-1)

            # calculate the NDVi :  (8A -4)/(8A + 4 + 1e-9)
            ndvi = (flatten_data[7 + i * 10] - flatten_data[2 + i * 10]) / (
                flatten_data[7 + i * 10] + flatten_data[2 + i * 10] + 1e-9
            )

            # calculate the Chlorophyll Index (CI)
            ci = (flatten_data[3 + i * 10] - flatten_data[4 + i * 10]) / (
                flatten_data[3 + i * 10] + flatten_data[4 + i * 10] + 1e-9
            )

            # append
            flatten_data.append(ci)
        """

        all_imgs = np.concatenate(flatten_data)
        return all_imgs

    def get_finaldata(self):
        feature_imgs = []
        for i in range(self.num):
            feature = self.all_features[i]
            # using old method: all features
            each_imgs = self.process_data(feature)
            if len(each_imgs) < len(self.bands) * 25:
                self.delete_index.append(i)
                continue

            feature_imgs.append(each_imgs)

        # get all flatten imgs in one class array(num, 750)
        self.newdata = np.stack(feature_imgs)
        """        assert self.newdata.shape == (
            self.num,
            len(self.bands) * 25 + self.num_newband * 25,
        ), "the shape of final data is not correct, != (num, 750)"
        """

        return self.newdata

    # get the name of this class
    def get_name(self):
        return self.name

    # get specific feature imgs
    def get_featureImgs_index(self, index: int):
        fea = self.all_features[index]
        imgs = self.process_data(fea)
        return imgs

    # get specific feature accoding to index
    def get_feature_index(self, index: int):
        feature = self.all_features[index]
        return feature

    # get the numbers of features
    def get_feature_num(self):
        return self.num

    # get the deleted null numbers of features
    def get_real_feature_num(self):
        return self.num - len(self.delete_index)

    # get the labels(np.array)
    def get_labels(self):
        labels = [x for i, x in enumerate(self.labels) if i not in self.delete_index]
        return labels

    def get_bands(self):
        return self.bands
