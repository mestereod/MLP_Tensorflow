import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

def writetxt(filename, string):
    f = open(filename + '.txt', 'w')
    f.write(string)
    f.close()

def openFile(filePath):
    '''
        filePath: an entire file path including the extension. For example: "MLP_Data/problemOR.csv"
    '''
    # just a function to open a csv inside our mlp class, just not to charge the user of knowing about pandas library
    data = pd.read_csv(filePath, header=None)
    return data

data = openFile('Data/problemAND.csv')
inputs = data.drop(data.columns[-1], axis=1).values
targets = data.drop(data.columns[:-1], axis=1).values
print(type(inputs))

s = tf.compat.v1.InteractiveSession()  # tf.compat.v1.InteractiveSession() is a way to run tensorflow model directly without instantiating a graph whenever we want to run a model.

## Defining various initialization parameters for 2-2-1 MLP model
num_classes = y_train.shape[1]
num_features = X_train.shape[1]
num_output = y_train.shape[1]
num_layers_0 = 2
num_layers_1 = 2
starter_learning_rate = 0.001
regularizer_rate = 0.1

print(num_classes)

# https://www.tutorialspoint.com/tensorflow/tensorflow_multi_layer_perceptron_learning.htm
# https://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/