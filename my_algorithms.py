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
import random
import copy

class My_ANN:
    """
    class Description: This class is intended to create/use an artificial neural
    network from scratch. It uses the signmoid activation derivative and backpropagration to
    train the network.

    Attributes
    ----------
    network
        a list of nodes

    Functions
    ---------
    """
    network = []
    train_data_ds = []
    train_label_ds = []
    test_data_ds = []
    test_label_ds = []

    def __init__(self, shape: tuple, train_data:list, train_label:list, test_data: list, test_label: list,) -> None:
        """
        Constructor Functions

        Parameters
        ----------
        shape
            The shape of the network
        train_data
        train_label
        test_data
        test_label
        """
        self.train_data_ds = train_data
        self.train_label_ds = train_label
        self.test_data_ds = test_data
        self.test_label_ds = test_label

        network_shape = list(shape)
        network_shape.insert(0, len(train_data[0]))

        self.network = []
        for index, num in enumerate(network_shape[1:], 1):
            layer = []
            for x in range(num):
                layer.append(Node(network_shape[index-1]))
            self.network.append(copy.deepcopy(layer))
        return None
    
    def test_ann(self) -> float:
        """
        Tests the ANN with the testing dataset and returns the accuracy
        """
        predictions = []
        for num, inputs in enumerate(self.test_data_ds):
            predicted_out = self.predict_var(inputs)
            if predicted_out == self.test_label_ds[num]:
                predictions.append([100, predicted_out, self.test_label_ds[num]])
            else:
                predictions.append([0, predicted_out, self.test_label_ds[num]])

        # calculating average accuracy for predictions list
        accuracy = 0
        for prediction in predictions:
            accuracy += prediction[0]

        # return accuracy
        return accuracy/len(predictions)
    # end test_ann

    def predict_var(self, var) -> str | int:
        """
        Makes a prediction when given a variable from a dataset
        """
        inputs = tuple(var)
        for layer in self.network:
            outputs = []
            for node in layer:
                outputs.append(node.get_output(inputs))
            inputs = tuple(outputs)

        # returning the index of the highest
        return outputs.index(max(outputs))
    
class Node:
    """
    class Description: Node for an ANN

    Attributes
    ----------
    inputs
    weights
    raw_out
    normalized_out

    Functions
    ---------
    get_output()
    """
    inputs = []
    weights = []
    raw_out = 0
    normalized_out = 0

    def __init__(self, num_inputs) -> None:
        """
        Constructor function
        
        Parameters
        ----------
        num_inputs
            The number of inputs which should equal number of weights needed
        """
        self.inputs = []
        # the +1 is for bias
        self.weights = numpy.random.random(num_inputs+1)
        self.raw_out = 0
        self.normalized_out = 0
        return None
    # end of __init__()

    def get_output(self, inputs: list[float]) -> float:
        """
        Returns the normalized output when given a list of inputs.

            Uses Sigmoid function to normalize output.

        Parameters
        ----------
        inputs
            list of inputs

        Returns
        -------
        float
            out
        """
        # checking that the number of inputs is correct
        if not (len(inputs)+1) == len(self.weights):
            print("\nError: Wrong number of inputs. Expected %s, got %s" % (len(self.weights), len(inputs)))

        # multiplying inputs by corresponding weights
        sum_output = 0
        for num, input in enumerate(inputs):
            sum_output += self.weights[num]*input

        # adding bias weights
        sum_output += self.weights[len(self.weights)-1]

        # storing data 
        self.inputs = inputs
        self.raw_out = sum_output
        self.normalized_out = 1/(1+math.e**(-1*sum_output)) # applying signmoid
        return self.normalized_out
    # end of get_output()

    def __str__(self) -> str:
        """
        returns a string equivalent of Node object
        """
        return str(self.weights)
        
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
    train_data_ds = []
    train_label_ds = []
    test_data_ds = []
    test_label_ds = []

    def __init__(self, train_data:list, train_label:list, test_data: list, test_label: list, k: int) -> None:
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