"""
Thomas Roberts
CS 4267(01) Machine Learning
Professor Ojo
April 11, 2024

Program Desciption: This program contains all the algorithms that will be
used in the main program to classify several datasets
"""

import numpy
import math

class My_KNN:
    """
    Description: This class holds the variables and functions related kNN 
    supervised sorting algorithm

    Attributes
    ----------
    train_ds
    test_ds

    Functions
    ---------
    test_knn()
    predict_var()
    euclidean_distance()
    """
    # k refers to the number of neighbors looked at
    k = 0
    my_classes = []
    train_data_ds = []
    train_label_ds = []
    test_data_ds = []
    test_label_ds = []

    def __init__(self, train_data:list, train_label:list, test_data: list, test_label: list, classes: list[int | str], k: int) -> None:
        """
        Constructor Function

        Parameters
        train_ds
            training dataset
        test_ds
            testing dataset
        """
        self.train_data_ds = train_data
        self.train_label_ds = train_label
        self.test_data_ds = test_data
        self.test_label_ds = test_label
        self.my_classes = classes
        self.k = k
        return None
    # end definition __init__

    def test_knn(self) -> float:
        """
        Tests the kNN model with the test dataset and returns the accuracy

        Returns
        --------
        float
            average accuracy of kNN model using test dataset
        """
        list_of_predictions = []

        # for every test variable, making a prediction
        for num, var in enumerate(self.test_data_ds):
            prediction = self.predict_var(var)
            if prediction == self.test_label_ds[num]:
                list_of_predictions.append([100, prediction, self.test_label_ds[num]])
            else:
                list_of_predictions.append([0, prediction, self.test_label_ds[num]])
        
        # calculating average accuracy for predictions list
        accuracy = 0
        for prediction in list_of_predictions:
            accuracy += prediction[0]

        # return accuracy
        return accuracy/len(list_of_predictions)

    def predict_var(self, var1: list) -> int | str:
        """
        Classifies a variable by k number of nearest neighbors

        Parameters
        -----------
        var1 
            list of inputs for 1 variable entry
        
        Returns
        ---------
        int | str
            The predicted class
        """
        list_of_distance = []

        # finding distance for each variable in training dataset
        for var in self.train_data_ds:
            list_of_distance.append(My_KNN.euclidean_distance(var, var1))

        # duplicating distance list and finding the lowest values
        sorted_list = list_of_distance.copy()
        sorted_list.sort()

        # creating a list that represents the classes of the nearest neighbors
        neighbors = [0 for var in self.my_classes]

        # for each value in lowest distance list, incrementing neighbor class list
        # (like taking a show of hands)
        for distance in sorted_list[:self.k]:
            c_index = list_of_distance.index(distance)
            c_class = self.train_label_ds[c_index]
            neighbors[c_class] += 1
        
        # returning the class of majority vote
        return neighbors.index(max(neighbors))

    def euclidean_distance(var1: list, var2: list) -> float:
        """
        Returns the euclidean distance between two variables 

            euclidean distance = sqrt(sum((var1[i]-var2[i])**2)), where i represents
            the attribute index

        Parameters
        ----------
        var1: list
        var2: list

        Returns
        -------
        euclidean_distance: float
        
        """
        # checking each var has same num attributes
        if not len(var1) == len(var2):
            print("Error calculated euclidean distance. Variable lists have different number of attributes")
            return -1
        # end if

        sum_dist = 0
        for i in range(len(var1)):
            sum_dist = sum_dist + (var1[i]-var2[i])**2

        return math.sqrt(sum_dist)
    # end definition