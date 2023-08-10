import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('penguins.csv')
df.replace({'species': {'Adelie': 1, 'Gentoo': 0, 'Chinstrap': -1},
            'gender': {'male': 1, 'female': 0}}, inplace=True)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['gender'] = df['gender'].astype(int)

train_c1 = df.iloc[:30]
test_c1 = df.iloc[30:50]
train_c2 = df.iloc[50:80]
test_c2 = df.iloc[80:100]
train_c3 = df.iloc[100:130]
test_c3 = df.iloc[130:]
# shuffled train
train_c1_c2 = pd.concat([train_c1, train_c2]).sample(frac=1)
train_c1_c3 = pd.concat([train_c1, train_c3]).sample(frac=1)
train_c2_c3 = pd.concat([train_c2, train_c3]).sample(frac=1)
# shuffled test
test_c1_c2 = pd.concat([test_c1, test_c2])
test_c1_c3 = pd.concat([test_c1, test_c3])
test_c2_c3 = pd.concat([test_c2, test_c3])
# TRAIN X
train_c1_c2_X = train_c1_c2.iloc[:, 1:]  # c1_c2_X
train_c1_c3_X = train_c1_c3.iloc[:, 1:]  # c1_c3_X
train_c2_c3_X = train_c2_c3.iloc[:, 1:]  # c1_c3_X
# TRAIN Y
train_c1_c2_Y = train_c1_c2.iloc[:, 0]  # c1_c2_Y
train_c1_c3_Y = train_c1_c3.iloc[:, 0]  # c1_c3_Y
train_c2_c3_Y = train_c2_c3.iloc[:, 0]  # c2_c3_Y

# TEST X
test_c1_c2_X = test_c1_c2.iloc[:, 1:]  # c1_c2_X
test_c1_c3_X = test_c1_c3.iloc[:, 1:]  # c1_c3_X
test_c2_c3_X = test_c2_c3.iloc[:, 1:]  # c1_c3_X

# TEST Y
test_c1_c2_Y = test_c1_c2.iloc[:, 0]  # c1_c2_Y
test_c1_c3_Y = test_c1_c3.iloc[:, 0]  # c1_c3_Y
test_c2_c3_Y = test_c2_c3.iloc[:, 0]  # c2_c3_Y


def plotting(c1, c2):
    plt.xlabel(c1)
    plt.ylabel(c2)
    plt.scatter(data=df[:50], x=c1, y=c2, c='r', label='Adelie')
    plt.scatter(data=df[50:100], x=c1, y=c2, c='g', label='Gentoo')
    plt.scatter(data=df[100:], x=c1, y=c2, c='b', label='Chinstrap')
    plt.legend(loc='upper right')
    plt.show()


def trainPerceptron(X_train, Y_train, Eta, number_of_epochs, Bias):
    X_train = np.array(X_train)
    #  convert the features list to array to get the net value for  each
    Y_train = np.array(Y_train)
    #  convert the actual train labels to array to get the net value for  each
    weights = np.random.random(X_train.shape[1])
    # create a random weights with size based on the number of columns in the features list
    for i in range(int(number_of_epochs)):
        # loop until reach the max number of epochs the user enter
        for j in range(0, len(X_train)):
            # loop around all the train features to get the net value for  each
            Net_Value = (np.dot(weights.T, X_train[j])) + Bias
            # apply the activation function (signum)
            if Net_Value >= 0:
                y = 1
            else:
                y = -1
            # compare between the actual train labels and the predicted one
            if y != Y_train[j]:
                # update the weights
                # (Y_train[j] - y) To calculate The loss L "Error"
                weights = weights + float(Eta) * (Y_train[j] - y) * X_train[j] + Bias
    # return the new weights
    return weights


def testPerceptron(X_test, W, Bias):
    results = []
    # list contains all the predicted labels
    X_test = np.array(X_test)
    #  convert the features list to array to get the net value for  each
    for i in range(len(X_test)):
        # loop around all the test features to get the net value for  each
        Net_Value = (np.dot(W.T, X_test[i])) + Bias
        # apply the activation function (signum)
        if Net_Value >= 0:
            y = 1
        else:
            y = -1
        results.append(y)
    # return the predicted labels
    return results


def confusion_matrix(Predicted, Actual):
    # Create 2 * 2 Array with zero value Then will loop through this array to add the actual values
    matrix = np.zeros((2, 2))
    # Loop until the last one predicted
    for i in range(len(Predicted)):
        # Compare Between The Predicted and The Actual Labels
        # Calculate Number of True Positive
        if int(Predicted[i]) == 1 and int(Actual[i]) == 1:
            # Add One in the first position in the matrix [0 ,0]
            matrix[0, 0] += 1
        # Calculate Number of True Negative
        elif int(Predicted[i]) == -1 and int(Actual[i]) == -1:
            # Add One in the Fourth position in the matrix [1 ,1]
            matrix[1, 1] += 1
        # Calculate Number of False Positive
        elif int(Predicted[i]) == 1 and int(Actual[i]) == -1:
            # Add One in the Second position in the matrix [1 ,0]
            matrix[1, 0] += 1
        # Calculate Number of False Negative
        elif int(Predicted[i]) == -1 and int(Actual[i]) == 1:
            # Add One in the Third position in the matrix [1 ,0]
            matrix[0, 1] += 1
    return matrix


def acc(Test_y, Test_y_P):
    # create initial value to the correct predicted
    correct = 0
    for i in range(len(Test_y)):
        # Compare Between The _predict and the actual one
        if Test_y[i] == Test_y_P[i]:
            # add one if The _predict and the actual one is the same
            correct += 1
    # return the percentage accuracy
    return correct / float(len(Test_y)) * 100.0
