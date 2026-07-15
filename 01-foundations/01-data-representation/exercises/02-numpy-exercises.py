import numpy as np


scalar = np.array(8) 

vector = np.array([1, 2, 3, 4])

matrix = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

tensor = np.zeros((2, 3, 4))


print("Scalar")
print("Value:", scalar)
print("Dimensions:", scalar.ndim)
print("Shape:", scalar.shape)
print("Data type:", scalar.dtype)

print("\nVector")
# Complete this section
print(vector)

print("\nMatrix")
# Complete this section
print(matrix)

print("\nTensor")
# Complete this section
print(tensor)