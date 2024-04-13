"""
Thomas Roberts
CS 4267(01) Machine Learning
Professor Ojo
April 11, 2024

Program Desciption: A dataset manager to handle dataset creation and
handling train/test splits.
"""

import numpy as np
import pandas as pd
import random

# Use sklearn inbuilt datasets:
from sklearn import datasets
from sklearn.model_selection import train_test_split as split


class Dataset_Manager:
    """
    class description: This class handles the dataset creation and splits
    related to this assignment.

    Attributes
    ----------
    iris_data
    iris_labels
    boston_data
    boston_labels
    cancer_data
    cancer_labels

    Functions
    ---------

    """
    # Iris Dataset
    iris_data = []
    iris_labels = []
    iris_classes = []

    # Boston dataset
    boston_data = []
    boston_labels = []

    # cancer dataset
    cancer_data = []
    cancer_labels = []

    def __init__(self) -> None:
        """
        Constructor function
        """

        # Iris dataset
        iris = datasets.load_iris()
        # retrieve the data
        self.iris_data = iris.data
        self.iris_labels = iris.target
        self.iris_classes = iris.target_names
        # min/max normalizing data
        self.iris_data = (self.iris_data-np.min(self.iris_data))/(np.max(self.iris_data)-np.min(self.iris_data))

        # boston dataset
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        self.boston_data = np.hstack(
            [raw_df.values[::2, :], raw_df.values[1::2, :2]])
        self.boston_labels = raw_df.values[1::2, 2]

        # cancer dataset
        cancer = datasets.load_breast_cancer()
        self.cancer_data = cancer.data
        self.cancer_labels = cancer.target

        return None
    # end definition __init__()]

    def get_iris_split(self, training_split: float = 0.7) -> tuple[list, list, list, list]:
        """
        Returns the iris dataset 

        Returns 
        -------
        train_data,
        train_labels,
        test_data,
        test_labels
        """
        # finding integer value that represents training_split num
        divisor = len(self.iris_data)*training_split
        divisor = round(divisor)

        # shuffles the data to add a degree of randomness. Data and Label correlation maintains integrity
        temp = list(zip(self.iris_data, self.iris_labels))
        random.shuffle(temp)
        self.iris_data, self.iris_labels = zip(*temp)

        # return train, test split as tuple
        dataset = tuple([self.iris_data[:divisor], self.iris_labels[:divisor],
                        self. iris_data[divisor:], self.iris_labels[divisor:]])
        return dataset
    
    def get_iris_classes(self) -> tuple:
        """
    
        """
        return tuple(self.iris_classes)
# end get_iris_classes