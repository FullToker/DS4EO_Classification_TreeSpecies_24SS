import numpy as np
from loader import File_geojson
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
data_path = [("../data/test/" + x + "_0528.geojson") for _, x in label_dict.items()]

# all classes
imgs = []
labels = []

for i in range(len(label_dict)):
    treeclass = File_geojson(data_path[i], label_dict)
    imgs.append(treeclass.get_finaldata())
    labels.append(treeclass.get_labels())
    # print(f"{treeclass.get_name()} : {treeclass.get_feature_num()}")


X = np.concatenate(imgs, axis=0)
y = np.concatenate(labels, axis=0)


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
    bands = imgs.reshape(30, 5, 5)
    flipped_bands = np.array([np.fliplr(block) for block in bands])
    flipped_bands = flipped_bands.reshape(30, 25).flatten()
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
    bands = imgs.reshape(30, 5, 5)
    flipped_bands = np.array([np.fliplr(block) for block in bands])
    flipped_bands = flipped_bands.reshape(30, 25).flatten()
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
    bands = imgs.reshape(30, 5, 5)
    rotated_90 = (
        np.array([np.rot90(block) for block in bands]).reshape(30, 25).flatten()
    )
    augm_ls.append(rotated_90)
    rotated_180 = (
        np.array([np.rot90(block, 2) for block in bands]).reshape(30, 25).flatten()
    )
    augm_ls.append(rotated_180)
    rotated_270 = (
        np.array([np.rot90(block, 3) for block in bands]).reshape(30, 25).flatten()
    )
    augm_ls.append(rotated_270)

    # flipped and rotate ---------///////----------------------------
    flipped_bands = np.array([np.fliplr(block) for block in bands])
    rotated_90 = (
        np.array([np.rot90(block) for block in flipped_bands]).reshape(30, 25).flatten()
    )
    augm_ls.append(rotated_90)
    rotated_180 = (
        np.array([np.rot90(block, 2) for block in flipped_bands])
        .reshape(30, 25)
        .flatten()
    )
    augm_ls.append(rotated_180)
    rotated_270 = (
        np.array([np.rot90(block, 3) for block in flipped_bands])
        .reshape(30, 25)
        .flatten()
    )
    augm_ls.append(rotated_270)

    augm_ls.append(flipped_bands.reshape(30, 25).flatten())
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
np.save("./data/X_augm.npy", X)
np.save("./data/y_augm.npy", y)

labels, counts = np.unique(y, return_counts=True)
for label, count in zip(labels, counts):
    print(f"{label_dict[label]}: {count} samples")


# Test through two rand images
"""
img1 = np.random.rand(5, 5)
img2 = np.random.rand(5, 5)


# 生成四个(2, 25)的输入
def generate_augmented_images(img1, img2):
    # 对两个图像进行各种翻转和旋转
    images = [
        img1.flatten(),
        img2.flatten(),
        np.fliplr(img1).flatten(),
        np.fliplr(img2).flatten(),
        np.rot90(img1).flatten(),
        np.rot90(img2).flatten(),
        np.rot90(img1, 3).flatten(),
        np.rot90(img2, 3).flatten(),
    ]
    # 将它们转换成(2, 25)的形式
    return [np.array(images[i : i + 2]) for i in range(0, len(images), 2)]


inputs = generate_augmented_images(img1, img2)


# 对四个(2, 25)进行PCA
def perform_pca(input_data):
    pca = PCA(n_components=1)
    pca.fit(input_data)
    first_principal_component = pca.components_[0]
    return first_principal_component


principal_components = [perform_pca(input_data) for input_data in inputs]

# 将四个input的第一主成分画在图中
x = np.arange(25)
plt.figure(figsize=(12, 12))

titles = [
    "Original Images",
    "Flipped Images",
    "90 Degree Rotated Images",
    "270 Degree Rotated Images",
]

for i, pca in enumerate(principal_components):
    plt.subplot(4, 1, i + 1)
    plt.bar(x, pca)
    plt.xlabel("Feature Index")
    plt.ylabel("First Principal Component")
    plt.title(titles[i])

plt.tight_layout()
plt.show()
"""
