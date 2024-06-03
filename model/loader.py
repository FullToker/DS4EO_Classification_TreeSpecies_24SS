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
            if len(img) != 5:
                print(img)
                img = np.zeros((5, 5))
            flatten_data.append(np.array(img).reshape(-1))
        all_imgs = np.concatenate(flatten_data)
        return all_imgs

    def get_finaldata(self):
        feature_imgs = []
        for i in range(self.num):
            feature = self.all_features[i]
            # using old method: all features
            # each_imgs = self.process_data(feature)

            # using  PCA Features to SFFS
            pca_features = Feature(feature)
            each_imgs = pca_features.__getPCA__()
            feature_imgs.append(each_imgs)

        # get all flatten imgs in one class array(num, 750)
        self.newdata = np.stack(feature_imgs)
        assert self.newdata.shape == (
            self.num,
            len(self.bands)
            * 3,  # if there is no PCA: * 25, otherwise: * PCA's n_components
        ), "the shape of final data is not correct, != (num, 750)"

        return self.newdata

    # get the name of this class
    def get_name(self):
        return self.name

    # get specific feature accoding to index
    def get_feature_index(self, index: int):
        feature = self.all_features[index]
        return feature

    # get the numbers of features
    def get_feature_num(self):
        return self.num

    # get the labels(np.array)
    def get_labels(self):
        return self.labels

    def get_bands(self):
        return self.bands


class Feature:
    def __init__(self, feature: dict):
        self.feature = feature
        self.pcaFeatures = []
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
        flatten_data = []
        # deal with "null":
        for band in self.bands:
            img = feature[band]
            if type(img) == type(None):
                img = np.zeros((5, 5))
            flatten_data.append(np.array(img).reshape(-1))
        self.filledImgs = np.concatenate(flatten_data)

    # reduce the dimension of each img: from 25 to 3 or 2
    def reduce_dimen(self):
        pca = PCA(n_components=3)
        reduced = []
        for i in range(self.filledImgs.shape[0]):
            band_img = self.filledImgs[i]
            pca.fit(band_img.reshape(1, -1))
            band_img_reduced = pca.transform(band_img.reshape(1, -1))
            reduced.append(band_img_reduced)
        self.pcaFeatures = np.concatenate(reduced)

    def __getFeature__(self):
        return self.feature

    def __getImgsShape__(self):
        return self.filledImgs.shape

    def __getPCA__(self):
        self.reduce_dimen()
        return self.pcaFeatures
