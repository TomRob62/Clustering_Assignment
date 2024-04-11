"""
Thomas Roberts
CS 4267(01) Machine Learning
Professor Ojo
April 11, 2024

Program Desciption: This program will execute the instructions list
in assignment#3_instructions. This includes implementing a clustering 
algorithm, logistic algorithm, and previous algorithm
"""
# Use sklearn inbuilt datasets:  
from sklearn import datasets 
# import some data 
iris = datasets.load_iris() 
#retrieve the data 
x = iris.data 
y = iris.target