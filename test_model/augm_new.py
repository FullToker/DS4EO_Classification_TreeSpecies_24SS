import numpy as np
from sklearn.model_selection import train_test_split

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
# original data distribution
"""
0  Picea abies: 1256
1 Fagus sylvatica: 945
2 Pinus sylvestris: 762
3 Quercus robur: 267  going to *4
4 Betula pendula: 452        going to *2
5 Quercus petraea: 178   going to *4
6 Fraxinus excelsior: 549    going to *2
7 Acer pseudoplatanus: 747
8 Sorbus aucuparia: 37       going to *8
9 Carpinus betulus: 504      going to *2
"""


X = np.load("./test_model/data/X_norm.npy")
y = np.load("./test_model/data/y_norm.npy")

print(f"Data shape: {X.shape}")
print(f"Label shape: {y.shape}")

# split the train set and val set
# augmentation only works on the Train set ------------------/////////////

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
# output the number of each class in train and val dataset
# number in train
print("number of train set")
for i in range(10):
    name = label_dict[i]
    print(f"{name}: {len(X_train[y_train==i])}, {len(y_train[y_train==i])}")
print()
print("number of val set")
for i in range(10):
    name = label_dict[i]
    print(f"{name}: {len(X_val[y_val==i])}, {len(y_val[y_val==i])}")

# save the val set to local
np.save("./test_model/data/X_val_norm_augm.npy", X_val)
np.save("./test_model/data/y_val_norm_augm.npy", y_val)


"""
number of train set
Picea abies: 846,
Fagus sylvatica: 669,
Pinus sylvestris: 536, 
Quercus robur: 191,  going to * 4
Betula pendula: 336,  going to *2
Quercus petraea: 122, going to * 4
Fraxinus excelsior: 393, going to *2
Acer pseudoplatanus: 518, 
Sorbus aucuparia: 27, going to *8
Carpinus betulus: 349, going to *2

number of val set
Picea abies: 410, 
Fagus sylvatica: 276, 
Pinus sylvestris: 226, 
Quercus robur: 76, 
Betula pendula: 116, 
Quercus petraea: 56, 
Fraxinus excelsior: 156,
Acer pseudoplatanus: 229, 
Sorbus aucuparia: 10, 
Carpinus betulus: 155, 
"""


def flipped(bands):
    flipped_bands = np.array([np.fliplr(block) for block in bands])
    flipped_bands = flipped_bands.reshape(30, 25).flatten()
    return flipped_bands


# angle = 90 180 270
def rotated(bands, angle: int):
    times = angle // 90
    rotated = (
        np.array([np.rot90(block, times) for block in bands]).reshape(30, 25).flatten()
    )
    return rotated


# do the augmentation for data, flag = 2(flipped); 4 (flipped + rotated 90);
# flag = 8 (flipped + rotated 90 180 270)
def augm_data(
    X_needAugm,
    label: int,
    flag: int,
):
    augm_x = []
    augm_y = []

    for row in X_needAugm:
        bands = row.reshape(30, 5, 5)
        if flag == 2:
            new_band1 = flipped(bands)
            augm_x.append(new_band1)

        if flag == 4:
            new_band1 = flipped(bands)
            new_band2 = rotated(bands, 90)

            new_band3 = rotated(new_band1.reshape(30, 5, 5), 90)
            augm_x.append(new_band1)
            augm_x.append(new_band2)
            augm_x.append(new_band3)

        if flag == 8:
            new_band1 = flipped(bands)
            new_band2 = rotated(bands, 90)
            new_band3 = rotated(bands, 180)
            new_band4 = rotated(bands, 270)

            new_band5 = rotated(new_band1.reshape(30, 5, 5), 90)
            new_band6 = rotated(new_band1.reshape(30, 5, 5), 180)
            new_band7 = rotated(new_band1.reshape(30, 5, 5), 270)
            augm_x.append(new_band1)
            augm_x.append(new_band2)
            augm_x.append(new_band3)
            augm_x.append(new_band4)
            augm_x.append(new_band5)
            augm_x.append(new_band6)
            augm_x.append(new_band7)

        # add (flag - 1) labels; for we add (flag - 1) imgs
        for _ in range(flag - 1):
            augm_y.append(label)

    augm_x = np.stack(augm_x)
    augm_y = np.stack(augm_y)
    return augm_x, augm_y


for i in range(10):
    x_process = None
    if i == 3 or i == 5:
        x_process = X_train[y_train == i]
        augm_x, augm_y = augm_data(x_process, i, flag=4)
        X_train = np.concatenate((X_train, augm_x), axis=0)
        y_train = np.concatenate((y_train, augm_y), axis=0)
        x_process = None
    if i == 4 or i == 6 or i == 9:
        x_process = X_train[y_train == i]
        augm_x, augm_y = augm_data(x_process, i, flag=2)
        X_train = np.concatenate((X_train, augm_x), axis=0)
        y_train = np.concatenate((y_train, augm_y), axis=0)
        x_process = None
    if i == 8:
        x_process = X_train[y_train == i]
        augm_x, augm_y = augm_data(x_process, i, flag=8)
        X_train = np.concatenate((X_train, augm_x), axis=0)
        y_train = np.concatenate((y_train, augm_y), axis=0)
        x_process = None


print("After Augm:")
print(f"x_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
for i in range(10):
    name = label_dict[i]
    print(f"{name}: {len(X_train[y_train==i])}, {len(y_train[y_train==i])}")

np.save("./test_model/data/X_train_norm_augm.npy", X_train)
np.save("./test_model/data/y_train_norm_augm.npy", y_train)


# old augmenation
"""
x shape: (8796, 750)
y shape: (8796,)
Picea abies: 1256, 1256
Fagus sylvatica: 945, 945
Pinus sylvestris: 762, 762
Quercus robur: 1068, 1068
Betula pendula: 904, 904
Quercus petraea: 712, 712
Fraxinus excelsior: 1098, 1098
Acer pseudoplatanus: 747, 747
Sorbus aucuparia: 296, 296
Carpinus betulus: 1008, 1008
"""

# strict right augmentatio, only for training set
"""
After Augm:
x_train shape: (6193, 750)
y_train shape: (6193,)
Picea abies: 846, 
Fagus sylvatica: 669, 
Pinus sylvestris: 536, 
Quercus robur: 764, 
Betula pendula: 672, 
Quercus petraea: 488, 
Fraxinus excelsior: 786, 
Acer pseudoplatanus: 518, 
Sorbus aucuparia: 216, 
Carpinus betulus: 698, 
"""
