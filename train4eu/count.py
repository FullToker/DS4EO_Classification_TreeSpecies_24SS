import numpy as np


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

X = np.load("../Dataset/X_all_outnull.npy")
y = np.load("../Dataset/y_all_outnull.npy")

for i in range(10):
    num = len(X[y == i])
    name = label_dict[i]
    print(f"{name}: {num}")
