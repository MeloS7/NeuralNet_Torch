from sklearn.datasets import fetch_california_housing
import numpy as np

# Generate a simple dataset for regression
def simple_dataset_regression(num_pairs=10):
    pair_list = []
    generated_X_values = set()
    while len(pair_list) < num_pairs:
        X = np.random.randint(1, 300)
        # Avoid duplicate X values
        if X in generated_X_values:
            continue
        y = 4 + 2 * X + np.random.randn(1)[0] * 30
        pair_list.append((X, y))
        generated_X_values.add(X)
    return pair_list


def load_house_dataset():
    housing = fetch_california_housing()
    return housing
