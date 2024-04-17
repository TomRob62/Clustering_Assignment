"""
Thomas Roberts
CS 4267(01) Machine Learning
Professor Ojo
April 11, 2024

Program Desciption: This program will execute the instructions list
in assignment#3_instructions. This includes implementing a clustering 
algorithm, logistic algorithm, and previous algorithm
"""
from dataset_manager import Dataset_Manager 
from my_algorithms import My_Cluster
from my_algorithms import My_Decision_Tree

# main algorithm

ds_manager = Dataset_Manager()

# part 1 - Clustering.
"""This clustering algorithm utilizes every data feature for a total of 12 features.
    The scatter plots are Feature vs VMED. Where the feature is shown as the title.
    The dataset and the starting centroids are **randomized**, so each execution will results
    in a different graph"""
boston_data, boston_labels = ds_manager.get_boston_whole()
print("\nClassifying Boston Dataset using k-means clustering algorithm with random starting centroids:\n")
cluster_obj = My_Cluster(boston_data, boston_labels, 50, 4)
cluster_obj.kmeans_clustering()
cluster_obj.display_clusters()

# part 2 - Decision Tree
iris_train_data, iris_train_label, iris_test_data, iris_test_label = ds_manager.get_iris_split()

# part 3 = Esemble Classifier
