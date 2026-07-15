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
print("Value:", vector)
print("Dimensions:", vector.ndim)
print("Shape:", vector.shape)
print("Data type:", vector.dtype)

print("\nMatrix")
# Complete this section
print("Value:", matrix)
print("Dimensions:", matrix.ndim)
print("Shape:", matrix.shape)
print("Data type:", matrix.dtype)

print("\nTensor")
# Complete this section
print("Value:", tensor)
print("Dimensions:", tensor.ndim)
print("Shape:", tensor.shape)
print("Data type:", tensor.dtype)