# main file for testing overfitting

import numpy as np
from data_loading import simple_dataset_regression
from utils import plot_data_list

def main():
    print("Hello, World!")
    dataset = simple_dataset_regression(15)
    print(dataset)
    plot_data_list(dataset)


if __name__ == "__main__":
    main()
