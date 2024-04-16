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
from copy import deepcopy
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
    get_iris_split()
    get_iris_classes()
    get_cancer_split()
    get_cancer_classes()
    get_boston_split()

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
    cancer_classes = []

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
        # min/max normalizing data
        self.boston_data = (self.boston_data-np.min(self.boston_data))/(np.max(self.boston_data)-np.min(self.boston_data))
        
        # cancer dataset
        cancer = datasets.load_breast_cancer()
        self.cancer_data = cancer.data
        self.cancer_labels = cancer.target
        self.cancer_classes = cancer.target_names
        # min/max normalizing data
        self.cancer_data = (self.cancer_data-np.min(self.cancer_data))/(np.max(self.cancer_data)-np.min(self.cancer_data))

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
    # end get_iris_split()
    
    def get_iris_classes(self) -> list[str]:
        """
        Returns the target names for Iris dataset
        """
        return self.iris_classes
    # end get_iris_classes()
    
    def get_cancer_split(self, training_split: float = 0.7) -> tuple[list, list, list, list]:
        """
        Returns the cancer dataset 

        Returns 
        -------
        train_data,
        train_labels,
        test_data,
        test_labels
        """
        # finding integer value that represents training_split num
        divisor = len(self.cancer_data)*training_split
        divisor = round(divisor)

        # shuffles the data to add a degree of randomness. Data and Label correlation maintains integrity
        temp = list(zip(self.cancer_data, self.cancer_labels))
        random.shuffle(temp)
        self.cancer_data, self.cancer_labels = zip(*temp)

        # return train, test split as tuple
        dataset = tuple([self.cancer_data[:divisor], self.cancer_labels[:divisor],
                        self. cancer_data[divisor:], self.cancer_labels[divisor:]])
        return dataset
    # end get_cancer_split()
    
    def get_cancer_classes(self) -> list[str]:
        """
        Returns the target names for cancer dataset
        """
        return self.cancer_classes
    # end get_cancer_classes()
    
    def get_boston_split(self, training_split: float = 0.7) -> tuple[list, list, list, list]:
        """
        Returns the boston dataset 

        Returns 
        -------
        train_data,
        train_labels,
        test_data,
        test_labels
        """
        # finding integer value that represents training_split num
        divisor = len(self.boston_data)*training_split
        divisor = round(divisor)

        # shuffles the data to add a degree of randomness. Data and Label correlation maintains integrity
        temp = list(zip(self.boston_data, self.boston_labels))
        random.shuffle(temp)
        self.boston_data, self.boston_labels = zip(*temp)

        # return train, test split as tuple
        dataset = tuple([self.boston_data[:divisor], self.boston_labels[:divisor],
                        self. boston_data[divisor:], self.boston_labels[divisor:]])
        return dataset
    # end get_boston_split()
    def get_boston_whole(self) -> dict:
        """
        Returns the boston dataset as a whole (instead of training/testing splits)
        """
        # shuffles the data to add a degree of randomness. Data and Label correlation maintains integrity
        temp = list(zip(self.boston_data, self.boston_labels))
        random.shuffle(temp)
        self.boston_data, self.boston_labels = zip(*temp)
        
        return self.boston_data, self.boston_labels
# end class Dataset Manager
