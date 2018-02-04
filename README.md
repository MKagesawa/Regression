# Regression & Gradient Descent
Housing Price Predictor (with 2 variables)

## Process
1. feature_normalization.py: Normalizes te data by subtracting the mean value of each feature and dividing its standard deviation
2. gradient_descent.py: train_model(normalized, max_iter) function finds the weight parameters for the model
3. gradient_descent.py: SGD(weights) fuction finds weight parameters for the SGD model
4. gradient_descent.py: plot() function makes graph of loss function with different number of iterations

## The training algorithm for Gradient Descent
```
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
```

## Acknowledgements
*Dataset used comes from http://cs229.stanford.edu/
