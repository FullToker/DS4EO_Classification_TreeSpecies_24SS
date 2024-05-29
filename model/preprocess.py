from loader import File_geojson
from scipy.linalg import orth

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
    ("../data/Dataset/test/" + x + "_0528.geojson") for _, x in label_dict.items()
]

carp = data_path[1]
carp = File_geojson(carp, label_dict)
imgs = carp.get_finaldata()
labels = carp.get_labels()
print(imgs.shape)
print(labels.shape)

img_num = imgs.shape[0]
bands = carp.get_bands()
lam = {
    "B2": 492.4,
    "B3": 559.8,
    "B4": 664.6,
    "B5": 704.1,
    "B6": 740.5,
    "B7": 782.8,
    "B8": 832.8,
    "B8A": 864.7,
    "B11": 1613.7,
    "B12": 2202.4,
    "B2_1": 492.4,
    "B3_1": 559.8,
    "B4_1": 664.6,
    "B5_1": 704.1,
    "B6_1": 740.5,
    "B7_1": 782.8,
    "B8_1": 832.8,
    "B8A_1": 864.7,
    "B11_1": 1613.7,
    "B12_1": 2202.4,
    "B2_2": 492.4,
    "B3_2": 559.8,
    "B4_2": 664.6,
    "B5_2": 704.1,
    "B6_2": 740.5,
    "B7_2": 782.8,
    "B8_2": 832.8,
    "B8A_2": 864.7,
    "B11_2": 1613.7,
    "B12_2": 2202.4,
}
first_feature = carp.get_feature_index(0)
# print(type(first_feature))
band_num = len(bands)
null_count = {
    "B2": 0,
    "B3": 0,
    "B4": 0,
    "B5": 0,
    "B6": 0,
    "B7": 0,
    "B8": 0,
    "B8A": 0,
    "B11": 0,
    "B12": 0,
    "B2_1": 0,
    "B3_1": 0,
    "B4_1": 0,
    "B5_1": 0,
    "B6_1": 0,
    "B7_1": 0,
    "B8_1": 0,
    "B8A_1": 0,
    "B11_1": 0,
    "B12_1": 0,
    "B2_2": 0,
    "B3_2": 0,
    "B4_2": 0,
    "B5_2": 0,
    "B6_2": 0,
    "B7_2": 0,
    "B8_2": 0,
    "B8A_2": 0,
    "B11_2": 0,
    "B12_2": 0,
}
for i in range(img_num):
    feature = carp.get_feature_index(i)
    for band in bands:
        if feature[band] == None:
            null_count[band] += 1
print(img_num)
print(null_count)
