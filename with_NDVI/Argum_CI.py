import numpy as np
from loader import File_geojson
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


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

X = np.load("./data/np_data/X_ndvi.npy")
y = np.load("./data/np_data/y_ndvi.npy")


# Augmentation test ----------------------------------//////////
augm_ls = []
augm_label = []
# for label is 3:
augm_class = X[y == 3]
tree = File_geojson(data_path[3], label_dict)
# iter every imgs
for i in range(len(augm_class)):
    imgs = tree.get_featureImgs_index(i)
    # iter every band
    bands = imgs.reshape(33, 5, 5)
    flipped_bands = np.array([np.fliplr(block) for block in bands])
    flipped_bands = flipped_bands.reshape(33, 25).flatten()
    augm_ls.append(flipped_bands)
    augm_label.append(3)


augm_X = np.stack(augm_ls)
augm_y = np.stack(augm_label)
X = np.concatenate((X, augm_X), axis=0)
y = np.concatenate((y, augm_y), axis=0)

# for label = 5:
augm_ls = []
augm_label = []
augm_class = X[y == 5]
tree = File_geojson(data_path[5], label_dict)
# iter every imgs
for i in range(len(augm_class)):
    imgs = tree.get_featureImgs_index(i)
    # iter every band
    bands = imgs.reshape(33, 5, 5)
    flipped_bands = np.array([np.fliplr(block) for block in bands])
    flipped_bands = flipped_bands.reshape(33, 25).flatten()
    augm_ls.append(flipped_bands)
    augm_label.append(5)
augm_X = np.stack(augm_ls)
augm_y = np.stack(augm_label)
X = np.concatenate((X, augm_X), axis=0)
y = np.concatenate((y, augm_y), axis=0)

# for label = 8
augm_ls = []
augm_label = []
augm_class = X[y == 8]
tree = File_geojson(data_path[8], label_dict)
add = 8
# iter every imgs
for i in range(len(augm_class)):
    imgs = tree.get_featureImgs_index(i)
    # iter every band
    bands = imgs.reshape(33, 5, 5)
    rotated_90 = (
        np.array([np.rot90(block) for block in bands]).reshape(33, 25).flatten()
    )
    augm_ls.append(rotated_90)
    rotated_180 = (
        np.array([np.rot90(block, 2) for block in bands]).reshape(33, 25).flatten()
    )
    augm_ls.append(rotated_180)
    rotated_270 = (
        np.array([np.rot90(block, 3) for block in bands]).reshape(33, 25).flatten()
    )
    augm_ls.append(rotated_270)

    # flipped and rotate ---------///////----------------------------
    flipped_bands = np.array([np.fliplr(block) for block in bands])
    rotated_90 = (
        np.array([np.rot90(block) for block in flipped_bands]).reshape(33, 25).flatten()
    )
    augm_ls.append(rotated_90)
    rotated_180 = (
        np.array([np.rot90(block, 2) for block in flipped_bands])
        .reshape(33, 25)
        .flatten()
    )
    augm_ls.append(rotated_180)
    rotated_270 = (
        np.array([np.rot90(block, 3) for block in flipped_bands])
        .reshape(33, 25)
        .flatten()
    )
    augm_ls.append(rotated_270)

    augm_ls.append(flipped_bands.reshape(33, 25).flatten())
    for _ in range(add - 1):
        augm_label.append(8)
augm_X = np.stack(augm_ls)
augm_y = np.stack(augm_label)
X = np.concatenate((X, augm_X), axis=0)
y = np.concatenate((y, augm_y), axis=0)
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")


# use the augm data  -----------------------------------------//////////////////////////
# print the labels with their num
np.save("./data/np_data/X_ci_augm.npy", X)
np.save("./data/np_data/y_ci_augm.npy", y)

# save as the torch.tensor
input = torch.tensor(X.reshape(-1, 33, 5, 5), dtype=torch.float32)
label = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(input, label)
train_data, test_data = random_split(
    dataset, [0.85, 0.15], generator=torch.Generator().manual_seed(42)
)
print(f"shape of train : {len(train_data)}")
print(f"shape of test : {len(test_data)}")
torch.save(train_data, "./encoder/save/train_argu.pth")
torch.save(test_data, "./encoder/save/test_argu.pth")


labels, counts = np.unique(y, return_counts=True)
for label, count in zip(labels, counts):
    print(f"{label_dict[label]}: {count} samples")
