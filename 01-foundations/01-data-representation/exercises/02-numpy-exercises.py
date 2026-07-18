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
# exp_2()


def exp_3():
    a = np.array([2, 4, 6])
    b = np.array([1, 3, 5])
    addition = a + b
    subtraction = a - b
    elementwise_multiplication = a * b
    division = a / b
    squared = np.square(a)
    dot_product_operator = a @ b
    dot_product_function = np.dot(a, b)

    print("\nExercise 3: Array operations")

    print("\nAddition:")
    print(addition)
    print("Shape:", addition.shape)

    print("\nSubtraction:")
    print(subtraction)
    print("Shape:", subtraction.shape)

    print("\nElement-wise multiplication:")
    print(elementwise_multiplication)
    print("Shape:", elementwise_multiplication.shape)

    print("\nDivision:")
    print(division)
    print("Shape:", division.shape)

    print("\nSquared values:")
    print(squared)
    print("Shape:", squared.shape)

    print("\nDot product using @:")
    print(dot_product_operator)
    print("Shape:", dot_product_operator.shape)

    print("\nDot product using np.dot:")
    print(dot_product_function)
    print("Shape:", dot_product_function.shape)



def exp_4():
    X = np.array([
        [2.0, 3.0],
        [4.0, 1.0],
        [5.0, 2.0],
        [1.0, 6.0],
    ])

    weights = np.array([0.7, 1.2])
    bias = 0.5

    weighted_scores = X @ weights
    predictions = weighted_scores + bias

    first_prediction_manual = (
        X[0, 0] * weights[0]
        + X[0, 1] * weights[1]
        + bias
    )

    print("\nExercise 4: Multiple linear predictions")

    print("\nDataset X:")
    print(X)
    print("Shape:", X.shape)

    print("\nWeights:")
    print(weights)
    print("Shape:", weights.shape)

    print("\nWeighted scores without bias:")
    print(weighted_scores)
    print("Shape:", weighted_scores.shape)

    print("\nFinal predictions:")
    print(predictions)
    print("Shape:", predictions.shape)

    print("\nFirst prediction calculated manually:")
    print(first_prediction_manual)

    print("\nBias contribution:")
    print(predictions - weighted_scores)
    




def exp_5():
    X = np.array([
        [20, 2000, 1],
        [25, 3000, 2],
        [30, 4000, 3],
        [35, 5000, 4],
        [40, 6000, 5],
    ])

    overall_mean = np.mean(X)
    feature_means = np.mean(X, axis=0) # by column
    observation_means = np.mean(X, axis=1) #by row
    feature_minimums = np.min(X, axis=0)
    feature_maximums = np.max(X, axis=0)
    feature_standard_deviations = np.std(X, axis=0)

    print("\nExercise 5: Statistics by axis")

    print("\nDataset:")
    print(X)
    print("Shape:", X.shape)

    print("\nOverall mean:")
    print(overall_mean)
    print("Shape:", overall_mean.shape)

    print("\nFeature means:")
    print(feature_means)
    print("Shape:", feature_means.shape)

    print("\nObservation means:")
    print(observation_means)
    print("Shape:", observation_means.shape)

    print("\nFeature minimums:")
    print(feature_minimums)
    print("Shape:", feature_minimums.shape)

    print("\nFeature maximums:")
    print(feature_maximums)
    print("Shape:", feature_maximums.shape)

    print("\nFeature standard deviations:")
    print(feature_standard_deviations)
    print("Shape:", feature_standard_deviations.shape)





exp_5()

