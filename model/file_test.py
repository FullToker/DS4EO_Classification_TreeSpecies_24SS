import numpy as np
from loader import File_geojson


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
    ("../data/Dataset/test/new_gee/" + x + "_0529.geojson")
    for _, x in label_dict.items()
]

# take Sorbus aucuparia as a test
sorbus = File_geojson(data_path[8], label_dict)
nums = sorbus.get_feature_num()
fea = sorbus.get_feature_index(0)
print(f"the nums of feature in sorbus is {nums}")

bands = [
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

for band in bands:
    img = fea[band]
    print(len(img))
