# main file for testing overfitting

import numpy as np
from data_loading import simple_dataset_regression


def main():
    print("Hello, World!")
    dataset = simple_dataset_regression()
    print(dataset)


if __name__ == "__main__":
    main()
