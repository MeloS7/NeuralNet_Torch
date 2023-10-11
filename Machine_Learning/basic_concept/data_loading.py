from sklearn.datasets import fetch_california_housing
import numpy as np

# Generate a simple dataset for regression
def simple_dataset_regression(num_paris=10):
    pair_list = []
    for _ in range(num_paris):
        X = np.random.rand(1, 50)
        y = 4 + 3 * X + np.random.randn(1, 100)
        pair_list.append((X, y))
    return pair_list


def load_house_dataset():
    housing = fetch_california_housing()
    return housing
