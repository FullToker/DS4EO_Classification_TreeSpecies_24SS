import numpy as np

X_test = np.load("./test/test_patches.npy")
y_test = np.load("./test/test_labels.npy")

print(X_test.shape)
print(y_test.shape)
print(y_test[0])
