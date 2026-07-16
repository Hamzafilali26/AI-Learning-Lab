import numpy as np

# Ex 1


def exp_1():
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
    print("Value:", vector)
    print("Dimensions:", vector.ndim)
    print("Shape:", vector.shape)
    print("Data type:", vector.dtype)

    print("\nMatrix")
    print("Value:", matrix)
    print("Dimensions:", matrix.ndim)
    print("Shape:", matrix.shape)
    print("Data type:", matrix.dtype)

    print("\nTensor")
    print("Value:", tensor)
    print("Dimensions:", tensor.ndim)
    print("Shape:", tensor.shape)
    print("Data type:", tensor.dtype)

# exp_1()
# Ex 2 
# Age; Monthly income; Number of purchases

def exp_2():
    X = np.array([
        [25, 5000, 2],
        [32, 7500, 5],
        [41, 9000, 7],
        [29, 6500, 4]
    ])

    second_customer = X[1]
    print(second_customer)
    ages = X[:,0]
    print(ages)
    incomes = X[:,1]
    print(incomes)
    first_two_customers = X[:2]
    print(first_two_customers)
    income_and_purchases = X[:,1:3]
    print(income_and_purchases)
    third_customer_purchases = X[2, 2]
    print(third_customer_purchases)
exp_2()



