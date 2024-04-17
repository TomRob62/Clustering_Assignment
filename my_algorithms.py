"""
Thomas Roberts
CS 4267(01) Machine Learning
Professor Ojo
April 11, 2024

Program Desciption: This program contains all the algorithms that will be
used in the main program to classify several datasets
"""
from random import randint
from matplotlib import pyplot as plt
import numpy
import math
import copy


class My_Cluster:
    """
    class Description: This class implements a k-means clustering algorithm

    Attributes
    -----------

    Functions
    --------
    """
    num_centroids = 0
    centroids = [numpy.ndarray]
    clusters = []
    cluster_labels = []

    data = []
    label = []

    def __init__(self, data: list, label: list, max_epoch: int, num_cent: int) -> None:
        """
        Constructor Function

        Parameters
        ----------
        train_data
        train_label
        test_data
        test_label
        num_cent
            number of centroids
        """
        self.data = data
        self.label = label
        self.max_epoch = max_epoch

        self.num_centroids = num_cent
        self.centroids = []
        self.clusters = [list() for x in range(num_cent)]
        self.cluster_labels = [list() for x in range(num_cent)]

        # randomly assigning n number of centeroids
        index_control = []
        for i in range(self.num_centroids):
            centroid_index = randint(0, (len(self.data)-1))
            # checking that no two centroids are the same
            while (index_control.__contains__(centroid_index)):
                centroid_index = randint(0, (len(self.data)-1))
            index_control.append(centroid_index)
            curr_centroid = copy.deepcopy(self.data[centroid_index])
            self.centroids.append(curr_centroid)
        return None
    # end definition __init__

    def kmeans_clustering(self) -> tuple[object]:
        """
        This method will cluster all the variables in the train dataset and
        return the clusters as a tuple such like:
            ([cluster 1], [cluster 2], ..., [cluster n])
        """
        old_centroids = []
        current_epoch = 0

        # begin clustering algorithm
        while (self.compare_centroids(old_centroids) == False and current_epoch < self.max_epoch):
            # cleaning clusters so we can add new variables
            self.clusters = [list() for x in range(self.num_centroids)]
            self.cluster_labels = [list() for x in range(self.num_centroids)]

            for num, variable in enumerate(self.data):
                distances = []
                for centroid in self.centroids:
                    distances.append(self.euclidean(centroid, variable))
                closest_index = distances.index(min(distances))
                self.cluster_labels[closest_index].append(self.label[num])
                self.clusters[closest_index].append(variable)
            # end for loop

            # updating stop conditions
            current_epoch += 1
            old_centroids = copy.deepcopy(self.centroids)
            for index in range(len(self.centroids)):
                self.calculate_new_centroid(index)
        # end while loop
        print("Developed Clusters in %s epoch." % current_epoch)
        return tuple(self.clusters)
    # end kmean_clustering

    # todo
    def display_clusters(self) -> None:
        """
        displays current clusters as a color coded scatterplot
        """
        # Plot the clusters obtained using k means
        fig = plt.figure()
        my_colors = ["red", "black", "blue", "green"]
        for num, my_color in enumerate(my_colors):
            features = [(12, 'LSTAT'), (5, 'RM'), (9, "Tax")]
            for i, feat in enumerate(features):
                my_cluster = []
                for var in self.clusters[num]:
                    my_cluster.append(var[feat[0]])
                # end for var loop
                plt.subplot(1, len(features), i+1)
                plt.scatter(my_cluster, self.cluster_labels[num], color=my_color)
                plt.title(feat[1])
                plt.xlabel(feat[1])
                plt.ylabel('MEDV')
            # end for feature loop
        # end for color loop
        plt.show()
        return None
    # end display_clusters

    def compare_centroids(self, old_centroids: numpy.ndarray) -> bool:
        """
        determines if two lists of centroids are the same. It it iterively compares
        the values of each corresponding centroid. This method is used as the 
        stopping condition for k mean clustering
        """
        if len(old_centroids) == 0:
            return False

        current_centroids = self.centroids
        for cent_index in range(len(current_centroids)):
            for val_index in range(len(current_centroids[cent_index])):
                old_value = old_centroids[cent_index][val_index]
                new_value = current_centroids[cent_index][val_index]
                if not old_value == new_value:
                    return False
            # end value for loop
        # end centroid for loop
        return True
    # end compare_centroids

    def euclidean(self, var1: list[float], var2: list[float]) -> float:
        """
        calculates the euclidean distance between two variables

        Parameters
        -----------
        var1
            iterable with real numbers
        var2 iterable with real numbers

        Returns
        -------
        float
        euclidean distance
        """
        if not len(var1) == len(var2):
            print(
                "\nError calculating euclidean distance. Object are not of the same length.")
        distance = 0
        for index in range(len(var1)):
            distance += (var1[index] - var2[index])**2

        return math.sqrt(distance)
    # end euclidean_distance()

    def calculate_new_centroid(self, index: int) -> None:
        """
        Calculates the new centroid values for a given cluster

        Parameters
        ----------
        index: int
            the index of the cluster/centroid being calculated
        """
        new_centroid = numpy.zeros(self.centroids[index].shape)
        for variable in self.clusters[index]:
            for num, value in enumerate(variable):
                new_centroid[num] += value
        new_centroid = new_centroid/(len(self.clusters[index]))
        self.centroids[index] = new_centroid
        return None
    # end calculate_new_centroid
# end class My_Cluster


class My_ANN:
    """
    class Description: This class is intended to create/use an artificial neural
    network from scratch. This ANN is used for multiclass categorization. This means 
    that the datasets must have an integer type label corresponding to the target 
    output.

    Attributes
    ----------
    network
        a list of nodes
    network_shape
    lr
        learning rate
    epoch
        max epoch for training
    train_data_ds
    train_label_ds
    test_data_ds
    test_label_ds

    Functions
    ---------
    test_ann()
    train_ann()
    predict_var()
        helper method that predicts the output for a single dataset variable
    """
    network = []     # holds the node objects
    network_shape = ()  # simple tuple to represent shape
    lr = 0  # learning rate
    epoch = 0  # max epoch
    # dataset variables
    train_data_ds = []
    train_label_ds = []
    test_data_ds = []
    test_label_ds = []

    def __init__(self, hidden_shape: tuple, num_out: int, train_data: list, train_label: list, test_data: list, test_label: list,) -> None:
        """
        Constructor Functions

        Parameters
        ----------
        hidden_shape
            The shape of the hidden layers
        num_out
            The number of target classes
        train_data
        train_label
        test_data
        test_label
        """
        self.train_data_ds = train_data
        self.train_label_ds = train_label
        self.test_data_ds = test_data
        self.test_label_ds = test_label

        # Inserting num of inputs and outputs in shape
        network_shape = []
        if hasattr(hidden_shape, '__iter__'):
            for num in hidden_shape:
                network_shape.append(num)
        else:
            network_shape.append(hidden_shape)
        network_shape.insert(0, len(train_data[0]))
        network_shape.append(num_out)
        self.network_shape = tuple(network_shape)

        # creating Node objects and appending them to my_ann network
        self.network = []
        for index, num in enumerate(network_shape[1:], 1):
            layer = []
            for x in range(num):
                layer.append(Node(network_shape[index-1]))
            self.network.append(copy.deepcopy(layer))
        return None

    def test_ann(self, dataset: int) -> float:
        """
        Tests the ANN with the selected dataset and returns the accuracy

            dataset 1 = training dataset
            dataset 2 = testing dataset

        Parameters
        ----------
        dataset: int
            the integer id assigned to the dataset
        """
        # bringing class variable to local method variable
        if dataset == 1:
            current_data = self.train_data_ds
            current_label = self.train_label_ds
        else:
            current_data = self.test_data_ds
            current_label = self.test_label_ds

        # storing predictions for accuracy metric
        predictions = []

        # iterating through each input, label entry in dataset
        for num, inputs in enumerate(current_data):
            # gets output for each output node
            outputs = self.predict_var(inputs)
            predicted_out = outputs.index(max(outputs))  # index of max output
            if predicted_out == current_label[num]:  # checking correctness
                predictions.append([100, predicted_out, current_label[num]])
            else:
                predictions.append([0, predicted_out, current_label[num]])

        # calculating average accuracy for predictions list
        accuracy = 0
        for prediction in predictions:
            accuracy += prediction[0]

        # return accuracy
        return accuracy/len(predictions)
    # end test_ann

    def train_ann(self, learning_rate: float, max_epoch: int) -> float:
        """
        This method trains the multiclass ANN with the training_dataset. It uses
        sum of squares loss function.

            Error = sum(0.5(actual - predicted)**2)

        Parameters
        -----------
        learning_rate
            recommended rate is 0.1
        max_epoch
            recommended rate is 1000
        """
        # conditions for stopping training
        training_accuracy = 0
        current_epoch = 0

        while training_accuracy < 99 and current_epoch < max_epoch:
            # one epoch
            training_accuracy = 0
            for train_index, train_var in enumerate(self.train_data_ds):
                # feed forward
                pred_out = self.predict_var(train_var)
                pred_index = pred_out.index(max(pred_out))

                # creating target out list
                target_out = [0 for num in pred_out]
                target_index = self.train_label_ds[train_index]
                target_out[target_index] = 1

                # adjusting training accuracy
                if pred_index == target_index:
                    training_accuracy += 100

                # begin backpropagation
                # calculating error = 0.5(actual - prediction)**2
                output_error = []
                for index in range(len(pred_out)):
                    output_error.append(
                        0.5*((target_out[index]-pred_out[index]**2)))

                # bringing output layer as local variable
                layer_index = len(self.network)-1
                layer = self.network[layer_index]

                # calculated weight adjustment for each node in output layer
                for num_node, out_node in enumerate(layer):
                    dE_dout = -1*(target_out[num_node] -
                                  out_node.normalized_out)
                    dout_dnet = out_node.normalized_out * \
                        (1-out_node.normalized_out)
                    out_node.adjust_weights(dE_dout, dout_dnet, learning_rate)

                # hidden layers
                layer_index -= 1
                while layer_index >= 0:  # iterated through each hidden layer in network
                    hidden_layer = self.network[layer_index]
                    # iterates through each node in current hidden layer
                    # and calculates weight adjustment
                    for hidden_index, hidden_node in enumerate(hidden_layer):
                        dE_dout = 0
                        for previous_node in self.network[layer_index+1]:
                            dE_dout += previous_node.get_derivatives(
                                hidden_index)
                        dout_dnet = hidden_node.normalized_out * \
                            (1-hidden_node.normalized_out)
                        hidden_node.adjust_weights(
                            dE_dout, dout_dnet, learning_rate)
                    # end node for loop
                    layer_index -= 1
                # end layer while loop
            # end epoch for loop

            # updating conditions to stop
            training_accuracy = training_accuracy/len(self.train_data_ds)
            current_epoch += 1
        # end training while loop
        return training_accuracy, current_epoch
    # end definition

    def predict_var(self, var) -> str | int:
        """
        Makes a prediction when given a variable from a dataset
        """
        inputs = tuple(var)
        for layer in self.network:
            outputs = []
            for node in layer:
                outputs.append(node.get_output(inputs))
            inputs = tuple(copy.copy(outputs))

        # returning the index of the highest
        return outputs
    # end of predict_var()

    def __str__(self) -> str:
        """
        Returns a string equivalents of My_ANN
        """
        ann_string = "\nANN of shape:" + str(self.network_shape)
        for num, layer in enumerate(self.network):
            ann_string += "\n\nLayer %s" % num
            for node in layer:
                ann_string += "\n" + str(node)
        return ann_string


class Node:
    """
    class Description: Node for an ANN

    Attributes
    ----------
    inputs
    weights
    raw_out
    normalized_out
    dE_dout
    dout_dnet

    Functions
    ---------
    get_derivates()
        helper method for backpropgation
    adjust_weights()
        helper method for backpropgation
    get_output()
        helper method for forward pass
    """
    inputs = []
    weights = []
    raw_out = 0
    normalized_out = 0
    # attributes for backpropagation
    dE_dout = 0
    dout_dnet = 0

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
        self.dE_dout = 0
        self.dout_dnet = 0
        return None
    # end of __init__()

    def get_derivatives(self, hidden_index) -> float:
        """
        returns multiplication of derivatives needed for backpropagation

        Parameters
        ----------
        hidden_index
            the index of the weight associated with node requesting the derivate
        """
        return (self.dE_dout*self.dout_dnet*self.weights[hidden_index])

    def adjust_weights(self, dE_dout: float, dout_dnet: float,  learning_rate: float) -> None:
        """
        Adjusts the weights of a node and returns the error of that node

        Parameters
        -----------
        learning rate: float
        previous: float
            error of the node prior to this
        """
        self.dE_dout = dE_dout
        self.dout_dnet = dout_dnet

        # adjusting weights
        for index in range(len(self.inputs)):
            self.weights[index] = self.weights[index] - \
                (learning_rate*dE_dout*dout_dnet*self.inputs[index])

        # adjusting bias
        self.weights[len(self.weights)-1] = self.weights[len(self.weights) -
                                                         1] - (learning_rate*dE_dout*dout_dnet)

        return None

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
            print("\nError: Wrong number of inputs. Expected %s, got %s" %
                  (len(self.weights), len(inputs)))

        # multiplying inputs by corresponding weights
        sum_output = 0
        for num, input in enumerate(inputs):
            sum_output += self.weights[num]*input

        # adding bias weights
        sum_output += self.weights[len(self.weights)-1]

        # storing data
        self.inputs = inputs
        self.raw_out = sum_output
        self.normalized_out = 1 / \
            (1+math.e**(-1*sum_output))  # applying signmoid
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

    def __init__(self, train_data: list, train_label: list, test_data: list, test_label: list, k: int) -> None:
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
                list_of_predictions.append(
                    [100, prediction, self.test_label_ds[num]])
            else:
                list_of_predictions.append(
                    [0, prediction, self.test_label_ds[num]])

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
            print(
                "Error calculated euclidean distance. Variable lists have different number of attributes")
            return -1
        # end if

        sum_dist = 0
        for i in range(len(var1)):
            sum_dist = sum_dist + (var1[i]-var2[i])**2

        return math.sqrt(sum_dist)
    # end definition
