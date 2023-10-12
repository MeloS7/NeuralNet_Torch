import matplotlib.pyplot as plt

# Plot data list
def plot_data_list(data_list):
    for data in data_list:
        X, y = data
        plt.scatter(X, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Predict y value from x value and parameters
def predict_y(x, num_params):
    y = 0
    for i in range(num_params+1): # include bias
        y += x ** i
    return y

# Loss functions
def quadratic_loss_function(y, y_hat):
    return (y_hat - y) ** 2

def absolute_loss_function(y, y_hat):
    return abs(y_hat - y)

# Calculate empirical risk
def empirical_risk(data_list, loss_function, num_params):
    total_loss = 0
    for data in data_list:
        X, y = data
        y_hat = predict_y(X, num_params)
        total_loss += loss_function(y, y_hat)
    return total_loss / len(data_list)

# Calculate empirical risk gradient
def empirical_risk_gradient(data_list, loss_function, num_params):
    total_loss_gradient = 0
    for data in data_list:
        X, y = data
        y_hat = predict_y(X, num_params)
        total_loss_gradient += (y_hat - y) * X
    return total_loss_gradient / len(data_list)