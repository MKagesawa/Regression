import matplotlib.pyplot as plt
from sympy import *
import pickle

norm_list_obj = open("norm_list.pkl", "rb")
norm_list = pickle.load(norm_list_obj)


def train_model(normalized, num_iter):

    # Initializing w0, w1, w2 parameters
    weights = [0.0, 0.0, 0.0]

    # Learning Rates
    alpha = [0.15, 0.01, 0.1, 0.3]

    x1 = [normalized[i][0] for i in range(len(normalized))]
    x2 = [normalized[i][1] for i in range(len(normalized))]
    y = [normalized[i][2] for i in range(len(normalized))]

    # Number of training samples
    m = len(x1)

    loss_function = sum([(weights[0] + weights[1] * x1[i] + weights[2] * x2[i] - y[i]) ** 2 for i in range(m)])
    loss_function = loss_function / (2*m)

    for i in range(num_iter):

        gradient0 = (sum([(weights[0] + weights[1] * x1[a] + weights[2] * x2[a] - y[a]) for a in range(m)])) * 1/m
        gradient1 = (sum([(weights[0] + weights[1] * x1[b] + weights[2] * x2[b] - y[b]) * x1[b] for b in range(m)])) * 1/m
        gradient2 = (sum([(weights[0] + weights[1] * x1[c] + weights[2] * x2[c] - y[c]) * x2[c] for c in range(m)])) * 1/m

        # Update weights
        weights[0] = weights[0] - (alpha[0] * gradient0)
        weights[1] = weights[1] - (alpha[0] * gradient1)
        weights[2] = weights[2] - (alpha[0] * gradient2)

        loss_function = sum([(weights[0] + weights[1] * x1[i] + weights[2] * x2[i] - y[i]) ** 2 for i in range(m)])
        loss_function = loss_function * 1/(2*m)

    return weights, loss_function


print(train_model(norm_list, 1000))
