<<<<<<< HEAD
# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librariesauxiliares
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

'''
firstly, we will define train data shape.
XOR train data has input X and output Y.

X is [4,2] shape like below,
[0, 0], [0, 1], [1, 0], [1, 1]

Y is [4,1] shape like below,
[[0], [1], [1], [0]]
'''
#neuronios
first_layer = 2
second_layer = 1

#X é o input
X = tf.placeholder(tf.float32, shape=[4,2])
#Y é o esperado
Y = tf.placeholder(tf.float32, shape=[4,1])

# we define first layer has two neurons taking two input values.  
W1 = tf.Variable(tf.random_uniform([2,2]))
# each neuron has one bias.
B1 = tf.Variable(tf.zeros([2]))
# First Layer's output is Z which is the sigmoid(W1 * X + B1)
Z = tf.sigmoid(tf.matmul(X, W1) + B1)


# we define second layer has one neurons taking two input values.  
W2 = tf.Variable(tf.random_uniform([2,1]))
# one neuron has one bias.
B2 = tf.Variable(tf.zeros([1]))
# Second Layer's output is Y_hat which is the sigmoid(W2 * Z + B2)
Y_hat = tf.sigmoid(tf.matmul(Z, W2) + B2)

# cross entropy
loss = tf.reduce_mean(-1*((Y*tf.log(Y_hat))+((1-Y)*tf.log(1.0-Y_hat))))

# Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# train data
train_X = [[0,0],[0,1],[1,0],[1,1]]
train_Y = [[0],[1],[1],[0]]

# initialize
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    print("train data: "+str(train_X))
    for i in range(20000):
        sess.run(train_step, feed_dict={X: train_X, Y: train_Y})
        if i % 5000 == 0:
            print('Epoch : ', i)
            print('Output : ', sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y}))
    
    print('Final Output : ', sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y}))

'''
def writetxt(filename, string):
    f = open(filename + '.txt', 'w')
    f.write(string)
    f.close()

def openFile(filePath):
        ##filePath: an entire file path including the extension. For example: "MLP_Data/problemOR.csv"
    # just a function to open a csv inside our mlp class, just not to charge the user of knowing about pandas library
    data = pd.read_csv(filePath, header=None)
    return data

def _and(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    global problem
    problem = 'AND'

    # making parameters log
    writetxt('AND_Parameters',
             'Input: ' + str(inputNumber) + '\nHidden: ' + str(hiddenNumber) + '\nOutput: ' + str(
                 outputNumber) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))

    mlp = MLP(inputNumber, hiddenNumber, outputNumber)

    # making initial weights log
    writetxt('AND_Initial_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemAND.csv')
    output = mlp.train(data, epochs, learningRate)

    inputs = data.drop(data.columns[-1], axis=1).values
    for inputvalue in inputs:
        print("Input: {} Predicted Output: {} Output: {}".format(inputvalue,1 if mlp.predict([inputvalue])[0] >= 0.5 else 0, mlp.predict([inputvalue])[0]))

    # making final weights log
    writetxt('AND_Final_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemAND.csv')
    plot(mlp,data,-2, 2, 6)



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
starter_learning_rate = 0.0001
regularizer_rate = 0.1

print(num_classes)

# https://dev.to/nexttech/introduction-to-multilayer-neural-networks-with-tensorflow-s-keras-api-39k6

# https://www.tutorialspoint.com/tensorflow/tensorflow_multi_layer_perceptron_learning.htm
# https://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
# https://github.com/smoreira/MultiLayerPerceptron/blob/master/mlp.py
'''
=======
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
>>>>>>> 4305bff6a6e2760f32923161bb1327ec8491cb0a
