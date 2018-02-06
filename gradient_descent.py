import matplotlib.pyplot as plt
from sympy import *
import pickle
import random

norm_list_obj = open("norm_list.pkl", "rb")
norm_list = pickle.load(norm_list_obj)


def train_model(normalized, max_iter):

    # Initializing w0, w1, w2 parameters
    weights = [0.0, 0.0, 0.0]

    # Learning Rates
    alpha = [0.15, 0.01, 0.1, 0.3, 0.05, 0.5]

    x1 = [normalized[i][0] for i in range(len(normalized))]
    x2 = [normalized[i][1] for i in range(len(normalized))]
    y = [normalized[i][2] for i in range(len(normalized))]

    # Number of training samples
    m = len(x1)

    loss_function = sum([(weights[0] + weights[1] * x1[i] + weights[2] * x2[i] - y[i]) ** 2 for i in range(m)])
    loss_function = loss_function / (2*m)

    for i in range(max_iter):

        gradient0 = (sum([(weights[0] + weights[1] * x1[a] + weights[2] * x2[a] - y[a]) for a in range(m)])) * 1/m
        gradient1 = (sum([(weights[0] + weights[1] * x1[b] + weights[2] * x2[b] - y[b]) * x1[b] for b in range(m)])) * 1/m
        gradient2 = (sum([(weights[0] + weights[1] * x1[c] + weights[2] * x2[c] - y[c]) * x2[c] for c in range(m)])) * 1/m

        # Update weights
        weights[0] = weights[0] - (alpha[2] * gradient0)
        weights[1] = weights[1] - (alpha[2] * gradient1)
        weights[2] = weights[2] - (alpha[2] * gradient2)

        loss_function = sum([(weights[0] + weights[1] * x1[i] + weights[2] * x2[i] - y[i]) ** 2 for i in range(m)])
        loss_function = loss_function * 1/(2*m)

    return weights, loss_function


# Randomized-mini-batch Stochastic Gradient Descent
def SGD_minibatch(weights):

    # Select a batch of data to train
    start = random.randint(0, len(norm_list) - 4)
    data_len = random.randint(4, len(norm_list) - start)

    data_batch = norm_list[start:start+data_len]
    print(start)
    print(data_len)
    print(data_batch)

    loss_function = 0

    # Each update of w uses only a single data point X^(i)
    for d in data_batch:
        g0 = weights[0] - 0.1 * (weights[0] + weights[1] * d[0] + weights[2] * d[1] - d[2])  # alpha set to 0.1
        g1 = weights[1] - 0.1 * (weights[0] + weights[1] * d[0] + weights[2] * d[1] - d[2]) * d[0]
        g2 = weights[2] - 0.1 * (weights[0] + weights[1] * d[0] + weights[2] * d[1] - d[2]) * d[1]

        # Update weights
        weights[0] = g0
        weights[1] = g1
        weights[2] = g2

        loss_function = sum([(weights[0] + weights[1] * norm_list[i][0] + weights[2] * norm_list[i][1] - norm_list[i][2]) ** 2 for i in range(len(norm_list))])
        loss_function = loss_function * 1/(2*data_len)
    print(loss_function)
    return weights


# print (SGD_minibatch(SGD_minibatch(SGD_minibatch([0.0, 0.0, 0.0]))))

# Stochastic Gradient Descent
def SGD(weights):

    data = norm_list

    # Shuffle data
    shuffled = []
    for i in range(len(norm_list)):
        element = random.choice(data)
        data.remove(element)
        shuffled.append(element)

    loss_function = 0

    # Each update of w uses only a single data point X^(i)
    for d in shuffled:
        g0 = weights[0] - 0.1 * (weights[0] + weights[1] * d[0] + weights[2] * d[1] - d[2])  # alpha set to 0.1
        g1 = weights[1] - 0.1 * (weights[0] + weights[1] * d[0] + weights[2] * d[1] - d[2]) * d[0]
        g2 = weights[2] - 0.1 * (weights[0] + weights[1] * d[0] + weights[2] * d[1] - d[2]) * d[1]

        # Update weights
        weights[0] = g0
        weights[1] = g1
        weights[2] = g2

        loss_function = sum([(weights[0] + weights[1] * shuffled[i][0] + weights[2] * shuffled[i][1] - shuffled[i][2])** 2 for i in range(len(shuffled))])
        loss_function = loss_function * 1/(2*len(shuffled))
    print(loss_function)
    return weights


print(SGD([341859.20577729295, 107831.74599088964, -2531.1739425848536]))


# Plot J(w) for with different alpha and number of iterations
def plot():
    J = []
    iterations = [10, 20, 30, 40, 50, 60, 70, 80]
    for i in iterations:
        J.append(train_model(norm_list, i)[1])
    plt.plot(iterations, J, 'ro')

    plt.show()

#Testing
#print(train_model(norm_list, 8000))