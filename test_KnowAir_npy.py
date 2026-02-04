import numpy as np

# read the numpy array from the .npy file
arr = np.load('KnowAir.npy')

# print the shape information
shape_info = arr.shape
print(shape_info)
