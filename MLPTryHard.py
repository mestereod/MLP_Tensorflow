<<<<<<< HEAD
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)


class MLP:

    def __init__(self, inputNumber, hiddenNumber, outputNumber):
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        ''' weights '''
      t  # a inputNumber+1(bias) x hiddenNumber matrix with random values. It already includes the bias weights
        # weightsInputToHidden[i][j] means the weight between the perceptron i of the hidden layer and the perceptron j of the input layer
        self.weightsInputToHidden = 2 * np.random.random((hiddenNumber, inputNumber + 1)) - 1
        # a hiddenNumber+1(bias) x outputNumber matrix with random values. It already includes the bias weights
        # weightsHiddenToOutput[i][j] means the weight between the perceptron i of the output layer and the perceptron j of the hidden layer
        self.weightsHiddenToOutput = 2 * np.random.random((outputNumber, hiddenNumber + 1)) - 1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedfoward(self, inputValues):
        # inputValues: a single line of input containg the attributes x1, x2, ..., xn

        # adding the bias as an inputValue to facilitate the calculus
        if len(inputValues) == self.inputNumber:
            inputValues = np.insert(inputValues, 0, 1)  # adding the value 1 in the index 0

        # feed values from input layer through hidden layer
        hiddenValues = np.zeros((self.hiddenNumber))  # an array for the values that will be calculated
        for i in range(self.hiddenNumber):  # for each hiddenNeuron
            summ = 0
            for j in range(len(inputValues)):  # each inputNeuron
                # linear combination of all the inputValues with their respective weight to a hidden perceptron i
                summ += inputValues[j] * self.weightsInputToHidden[i][j] 
            hiddenValues[i] = self.sigmoid(summ)  # applying the step function to the summation

        # adding the bias as a hiddenValue to facilitate the calculus
        # this is a development preference, the weights already contain the bias's weights, then, the bias comes as an input
        # this have to be treated at the backpropagation because the bias node does not propagate an error because the previous layer is not connected to it
        if len(hiddenValues) == self.hiddenNumber:
            # adding the value 1 in the index 0
            hiddenValues = np.insert(hiddenValues, 0, 1)

        # feed values from hidden layer through output layer
        outputValues = np.zeros((self.outputNumber))
        for i in range(self.outputNumber):  # for each outputNeuron
            summ = 0
            for j in range(len(hiddenValues)):  # each hiddenNeuron
                # linear combination of all the hiddenValues with their respective weight to an output perceptron i
                summ += hiddenValues[j] * self.weightsHiddenToOutput[i][j]
            outputValues[i] = self.sigmoid(summ)  # applying the step function to the summation

        return outputValues, hiddenValues, inputValues

    def backpropagation(self, targetValues, inputValues, learningRate):

        # executing the feedfoward and receiving all of the values of output from each neuron on each layer
        # so outputValues are the values of the output neurons, hidden values are the values of the hidden neurons, and so on
        (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)

        # setting a matrix for calculate the output errors, and calculating it
        outputErrors = np.zeros((self.outputNumber))
        # for each neuron at the output layer we calculate the difference of the target value and the output neuron value
        # also we multiply that difference with the derivative of the step function applied to the respective output neuron value
        for i in range(self.outputNumber):
            outputErrors[i] = (targetValues[i] - outputValues[i]) * self.sigmoid_derivative(outputValues[i])

        # getting the delta (change) of the weights from the hidden-layer through output-layer, considering the errors of the output layer calculated above
        deltasHiddenToOutput = np.zeros((self.outputNumber, self.hiddenNumber + 1))
        for i in range(self.outputNumber):
            for j in range(self.hiddenNumber + 1):
                # for each weight that are connected to a output neuron i we store the change for that weight in a deltas array
                # this change is calculated by the product of the learning rate, the error of the neuron i, and the value that following that weight "caused" the error
                deltasHiddenToOutput[i][j] = learningRate * outputErrors[i] * hiddenValues[j]

        # setting a matrix for calculate the hidden errors, and calculating it using the errors got by the output layer
        # here we have to be cautelous, the errors are only for the neurons, not the bias, but our weights matrix have the bias's weights
        # so we have to ignore the bias's weights in this step of backpropagating the error to previous nodes
        # this is why we iterate from index 1 through hiddenNumber+1
        hiddenErrors = np.zeros((self.hiddenNumber))
        for i in range(1, self.hiddenNumber + 1):
            summ = 0
            # calculating the linear combination of all the output layer neuron errors with the respective weight that leads to that error
            # for example an error of a hidden neuron i will be the summation of the product of all the errors of output neurons with the weight that connect the hidden neuron i with these neurons
            for j in range(self.outputNumber):
                summ += outputErrors[j] * self.weightsHiddenToOutput[j][i]
            # because of a neural network is a bunch of nested functions, the derivative in order to find the minimum error implies in several chain rules
            # so every error propagated has a value that is multiplied with the derivative of the step function applied to the value of the node that receives the error
            hiddenErrors[i - 1] = self.sigmoid_derivative(hiddenValues[i]) * summ

            # getting the delta (change) of the weights from the input-layer through the hidden-layer, considering the errors of the hidden layer calculated above
        deltasInputToHidden = np.zeros((self.hiddenNumber, self.inputNumber + 1))
        for i in range(self.hiddenNumber):
            for j in range(self.inputNumber + 1):
                # for each weight from the input layer that are connected to a hidden neuron i we store the change for that weight a the deltas array
                # this change is calculated by the product of the learning rate, the error of the neuron i, and the value that following that weight "caused" the error
                deltasInputToHidden[i][j] = learningRate * hiddenErrors[i] * inputValues[j]

        # updating the weights
        # only and finally, adding the deltas(changes) to the current weights
        for i in range(len(self.weightsHiddenToOutput)):
            for j in range(len(self.weightsHiddenToOutput[i])):
                self.weightsHiddenToOutput[i][j] += deltasHiddenToOutput[i][j]

        for i in range(len(self.weightsInputToHidden)):
            for j in range(len(self.weightsInputToHidden[i])):
                self.weightsInputToHidden[i][j] += deltasInputToHidden[i][j]

    def train(self, trainSet, epochs=1000, learningRate=1, learningRateMultiplierPerEpoch=1):
        '''
            trainSet: a pandas dataframe with the values for training
        '''
        # data treatment, creating a numpy array for only the inputs, and another for only the targets
        inputs = trainSet.drop(trainSet.columns[-1], axis=1).values
        targets = trainSet.drop(trainSet.columns[:-1], axis=1).values

        errorPerEpoch = ''
        outputPerEpoch = ''
        for epoch in range(epochs):
            # for each epoch we iterate over all the input and targets of a specific case and send it to the backpropagation function
            for inputValues, targetValues in zip(inputs, targets):
                # this condition is a data treatment for targets with more than one value, for example targets that are arrays
                if type(targetValues[0]) == type(np.array((2))):
                    targetValues = targetValues[0]

                # making error and output per epoch log
                (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)
                errorPerEpoch += 'Epoch: ' + str(epoch + 1) + '\nInput: ' + str(inputValues) + '\nError: ' + \
                                 str((targetValues - outputValues) * self.sigmoid_derivative(outputValues)) + '\n'
                outputPerEpoch += 'Epoch: ' + str(epoch + 1) + '\nInput: ' + str(inputValues) + '\nOutput: ' + str(
                    outputValues) + '\n'

                self.backpropagation(targetValues, inputValues, learningRate)
                # updating the learning rate according to the multiplier, if the multiplier is 1, we can assume that our learning rate is static
                learningRate *= learningRateMultiplierPerEpoch 

        # making error and output per epoch log
        writetxt(problem + '_Errors_Per_Epoch', errorPerEpoch)
        writetxt(problem + '_Outputs_Per_Epoch', outputPerEpoch)

        # at the end it returns the a prediction of the inputs that were used to train the model
        # what is expected is that the predictions match with the target values of each input case
        return self.predict(inputs)

    def predict(self, inputs):
        output = []
        for inputValues in inputs:
            # calling feed foward for each input case and receiving the output for each case, that is stored in the output list
            (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)
            output.append(outputValues)
        # at the end we return all the outputs of our model in a list
        return output

    def openFile(self, filePath):
        '''
            filePath: an entire file path including the extension. For example: "MLP_Data/problemOR.csv"
        '''
        # just a function to open a csv inside our mlp class, just not to charge the user of knowing about pandas library
        data = pd.read_csv(filePath, header=None)
        return data

# global variable
problem = ''

def writetxt(filename, string):
    f = open(filename + '.txt', 'w')
    f.write(string)
    f.close()

def plot(mlp,dataframe, start, end,scale):

    global problem

    x = []
    for i in range(start*scale, end*scale):
        for j in range(start*scale,end*scale):
            x.append([i/scale,j/scale])
    x = np.array(x)
    #print(x)
    colors = []
    inputs = dataframe.drop(dataframe.columns[-1], axis=1)
    targets = dataframe.drop(dataframe.columns[:-1], axis=1).values
    for y in targets:
        colors.append('red') if y == 0 else colors.append('green')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(inputs[inputs.columns[0]], inputs[inputs.columns[1]],targets,color=colors)
    y = mlp.predict(x)
    y = [y[i][0] for i in range(len(y))]
    ax.plot_trisurf(x[:,0],x[:,1], y)
    plt.suptitle(problem)
    plt.show()

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


def _or(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    global problem
    problem = 'OR'

    # making parameters log
    writetxt('OR_Parameters',
             'Input: ' + str(inputNumber) + '\nHidden: ' + str(hiddenNumber) + '\nOutput: ' + str(
                 outputNumber) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))

    mlp = MLP(inputNumber, hiddenNumber, outputNumber)

    # making initial weights log
    writetxt('OR_Initial_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemOR.csv')
    output = mlp.train(data, epochs, learningRate)

    inputs = data.drop(data.columns[-1], axis=1).values
    for inputvalue in inputs:
        print("Input: {} Predicted Output: {} Output: {}".format(inputvalue,1 if mlp.predict([inputvalue])[0] >= 0.5 else 0, mlp.predict([inputvalue])[0]))

    # making final weights log
    writetxt('OR_Final_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemOR.csv')
    plot(mlp, data, -2, 2, 6)


def _xor(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    global problem
    problem = 'XOR'

    # making parameters log
    writetxt('XOR_Parameters',
             'Input: ' + str(inputNumber) + '\nHidden: ' + str(hiddenNumber) + '\nOutput: ' + str(
                 outputNumber) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))

    mlp = MLP(inputNumber, hiddenNumber, outputNumber)

    # making initial weights log
    writetxt('XOR_Initial_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemXOR.csv')
    output = mlp.train(data, epochs, learningRate)

    inputs = data.drop(data.columns[-1], axis=1).values
    for inputvalue in inputs:
       print("Input: {} Predicted Output: {} Output: {}".format(inputvalue,1 if mlp.predict([inputvalue])[0] >= 0.5 else 0, mlp.predict([inputvalue])[0]))

    # making final weights log
    writetxt('XOR_Final_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemXOR.csv')
    plot(mlp, data, -2, 2, 6)


def _characters(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    global problem
    problem = 'CHAR'

    # making parameters log
    writetxt('CHAR_Parameters',
             'Input: ' + str(inputNumber) + '\nHidden: ' + str(hiddenNumber) + '\nOutput: ' + str(
                 outputNumber) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))

    mlp = MLP(inputNumber, hiddenNumber, outputNumber)

    # making initial weights log
    writetxt('CHAR_Initial_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/caracteres-limpo.csv')
    # mapping letters to array of values
    letters_map = {'A': np.array([1, 0, 0, 0, 0, 0, 0]),
                   'B': np.array([0, 1, 0, 0, 0, 0, 0]),
                   'C': np.array([0, 0, 1, 0, 0, 0, 0]),
                   'D': np.array([0, 0, 0, 1, 0, 0, 0]),
                   'E': np.array([0, 0, 0, 0, 1, 0, 0]),
                   'J': np.array([0, 0, 0, 0, 0, 1, 0]),
                   'K': np.array([0, 0, 0, 0, 0, 0, 1])}
    print('Testing with a clean dataframe:')
    data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: letters_map[x])

    # training and predict
    output = mlp.train(data, epochs, learningRate)
    inputs = data.drop(data.columns[-1], axis=1).values
    # making final weights log
    writetxt('CHAR_Final_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    letters = list(letters_map.keys())

    for inputValues, outputValues, targetValues in zip(inputs, output, data[data.columns[-1]].values):
        i = max(range(len(targetValues)), key=lambda x: targetValues[x])
        print('Expected Output:', letters[i], letters_map[letters[i]], end=' ')
        i = max(range(len(outputValues)), key=lambda x: outputValues[x])
        print('Output:', letters[i], letters_map[letters[i]], outputValues)

    ''' Noisy Characters '''
    print('\nTesting with a dataframe containing messy pixels:')
    # testing on a messy pixels characters dataframe
    data = mlp.openFile('Data/caracteres-ruido.csv')
    # mapping letters to array of values
    data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: letters_map[x])

    # data treatment
    inputs = data.drop(data.columns[-1], axis=1).values
    output = mlp.predict(inputs)

    # predict
    for outputValues, targetValues in zip(output, data[data.columns[-1]].values):
        i = max(range(len(targetValues)), key=lambda x: targetValues[x])
        print('Expected Output:', letters[i], letters_map[letters[i]], end=' ')
        i = max(range(len(outputValues)), key=lambda x: outputValues[x])
        print('Output:', letters[i], letters_map[letters[i]], outputValues)


_and(2,4,1,100,0.7)
#_or(2,4,1,100,0.5)
#_xor(2,4,1,1000,0.5)
=======
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)


class MLP:

    def __init__(self, inputNumber, hiddenNumber, outputNumber):
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        ''' weights '''
        # a inputNumber+1(bias) x hiddenNumber matrix with random values. It already includes the bias weights
        # weightsInputToHidden[i][j] means the weight between the perceptron i of the hidden layer and the perceptron j of the input layer
        self.weightsInputToHidden = 2 * np.random.random((hiddenNumber, inputNumber + 1)) - 1
        # a hiddenNumber+1(bias) x outputNumber matrix with random values. It already includes the bias weights
        # weightsHiddenToOutput[i][j] means the weight between the perceptron i of the output layer and the perceptron j of the hidden layer
        self.weightsHiddenToOutput = 2 * np.random.random((outputNumber, hiddenNumber + 1)) - 1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedfoward(self, inputValues):
        # inputValues: a single line of input containg the attributes x1, x2, ..., xn

        # adding the bias as an inputValue to facilitate the calculus
        if len(inputValues) == self.inputNumber:
            inputValues = np.insert(inputValues, 0, 1)  # adding the value 1 in the index 0

        # feed values from input layer through hidden layer
        hiddenValues = np.zeros((self.hiddenNumber))  # an array for the values that will be calculated
        for i in range(self.hiddenNumber):  # for each hiddenNeuron
            summ = 0
            for j in range(len(inputValues)):  # each inputNeuron
                # linear combination of all the inputValues with their respective weight to a hidden perceptron i
                summ += inputValues[j] * self.weightsInputToHidden[i][j] 
            hiddenValues[i] = self.sigmoid(summ)  # applying the step function to the summation

        # adding the bias as a hiddenValue to facilitate the calculus
        # this is a development preference, the weights already contain the bias's weights, then, the bias comes as an input
        # this have to be treated at the backpropagation because the bias node does not propagate an error because the previous layer is not connected to it
        if len(hiddenValues) == self.hiddenNumber:
            # adding the value 1 in the index 0
            hiddenValues = np.insert(hiddenValues, 0, 1)

        # feed values from hidden layer through output layer
        outputValues = np.zeros((self.outputNumber))
        for i in range(self.outputNumber):  # for each outputNeuron
            summ = 0
            for j in range(len(hiddenValues)):  # each hiddenNeuron
                # linear combination of all the hiddenValues with their respective weight to an output perceptron i
                summ += hiddenValues[j] * self.weightsHiddenToOutput[i][j]
            outputValues[i] = self.sigmoid(summ)  # applying the step function to the summation

        return outputValues, hiddenValues, inputValues

    def backpropagation(self, targetValues, inputValues, learningRate):

        # executing the feedfoward and receiving all of the values of output from each neuron on each layer
        # so outputValues are the values of the output neurons, hidden values are the values of the hidden neurons, and so on
        (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)

        # setting a matrix for calculate the output errors, and calculating it
        outputErrors = np.zeros((self.outputNumber))
        # for each neuron at the output layer we calculate the difference of the target value and the output neuron value
        # also we multiply that difference with the derivative of the step function applied to the respective output neuron value
        for i in range(self.outputNumber):
            outputErrors[i] = (targetValues[i] - outputValues[i]) * self.sigmoid_derivative(outputValues[i])

        # getting the delta (change) of the weights from the hidden-layer through output-layer, considering the errors of the output layer calculated above
        deltasHiddenToOutput = np.zeros((self.outputNumber, self.hiddenNumber + 1))
        for i in range(self.outputNumber):
            for j in range(self.hiddenNumber + 1):
                # for each weight that are connected to a output neuron i we store the change for that weight in a deltas array
                # this change is calculated by the product of the learning rate, the error of the neuron i, and the value that following that weight "caused" the error
                deltasHiddenToOutput[i][j] = learningRate * outputErrors[i] * hiddenValues[j]

        # setting a matrix for calculate the hidden errors, and calculating it using the errors got by the output layer
        # here we have to be cautelous, the errors are only for the neurons, not the bias, but our weights matrix have the bias's weights
        # so we have to ignore the bias's weights in this step of backpropagating the error to previous nodes
        # this is why we iterate from index 1 through hiddenNumber+1
        hiddenErrors = np.zeros((self.hiddenNumber))
        for i in range(1, self.hiddenNumber + 1):
            summ = 0
            # calculating the linear combination of all the output layer neuron errors with the respective weight that leads to that error
            # for example an error of a hidden neuron i will be the summation of the product of all the errors of output neurons with the weight that connect the hidden neuron i with these neurons
            for j in range(self.outputNumber):
                summ += outputErrors[j] * self.weightsHiddenToOutput[j][i]
            # because of a neural network is a bunch of nested functions, the derivative in order to find the minimum error implies in several chain rules
            # so every error propagated has a value that is multiplied with the derivative of the step function applied to the value of the node that receives the error
            hiddenErrors[i - 1] = self.sigmoid_derivative(hiddenValues[i]) * summ

            # getting the delta (change) of the weights from the input-layer through the hidden-layer, considering the errors of the hidden layer calculated above
        deltasInputToHidden = np.zeros((self.hiddenNumber, self.inputNumber + 1))
        for i in range(self.hiddenNumber):
            for j in range(self.inputNumber + 1):
                # for each weight from the input layer that are connected to a hidden neuron i we store the change for that weight a the deltas array
                # this change is calculated by the product of the learning rate, the error of the neuron i, and the value that following that weight "caused" the error
                deltasInputToHidden[i][j] = learningRate * hiddenErrors[i] * inputValues[j]

        # updating the weights
        # only and finally, adding the deltas(changes) to the current weights
        for i in range(len(self.weightsHiddenToOutput)):
            for j in range(len(self.weightsHiddenToOutput[i])):
                self.weightsHiddenToOutput[i][j] += deltasHiddenToOutput[i][j]

        for i in range(len(self.weightsInputToHidden)):
            for j in range(len(self.weightsInputToHidden[i])):
                self.weightsInputToHidden[i][j] += deltasInputToHidden[i][j]

    def train(self, trainSet, epochs=1000, learningRate=1, learningRateMultiplierPerEpoch=1):
        '''
            trainSet: a pandas dataframe with the values for training
        '''
        # data treatment, creating a numpy array for only the inputs, and another for only the targets
        inputs = trainSet.drop(trainSet.columns[-1], axis=1).values
        targets = trainSet.drop(trainSet.columns[:-1], axis=1).values

        errorPerEpoch = ''
        outputPerEpoch = ''
        for epoch in range(epochs):
            # for each epoch we iterate over all the input and targets of a specific case and send it to the backpropagation function
            for inputValues, targetValues in zip(inputs, targets):
                # this condition is a data treatment for targets with more than one value, for example targets that are arrays
                if type(targetValues[0]) == type(np.array((2))):
                    targetValues = targetValues[0]

                # making error and output per epoch log
                (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)
                errorPerEpoch += 'Epoch: ' + str(epoch + 1) + '\nInput: ' + str(inputValues) + '\nError: ' + \
                                 str((targetValues - outputValues) * self.sigmoid_derivative(outputValues)) + '\n'
                outputPerEpoch += 'Epoch: ' + str(epoch + 1) + '\nInput: ' + str(inputValues) + '\nOutput: ' + str(
                    outputValues) + '\n'

                self.backpropagation(targetValues, inputValues, learningRate)
                # updating the learning rate according to the multiplier, if the multiplier is 1, we can assume that our learning rate is static
                learningRate *= learningRateMultiplierPerEpoch 

        # making error and output per epoch log
        writetxt(problem + '_Errors_Per_Epoch', errorPerEpoch)
        writetxt(problem + '_Outputs_Per_Epoch', outputPerEpoch)

        # at the end it returns the a prediction of the inputs that were used to train the model
        # what is expected is that the predictions match with the target values of each input case
        return self.predict(inputs)

    def predict(self, inputs):
        output = []
        for inputValues in inputs:
            # calling feed foward for each input case and receiving the output for each case, that is stored in the output list
            (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)
            output.append(outputValues)
        # at the end we return all the outputs of our model in a list
        return output

    def openFile(self, filePath):
        '''
            filePath: an entire file path including the extension. For example: "MLP_Data/problemOR.csv"
        '''
        # just a function to open a csv inside our mlp class, just not to charge the user of knowing about pandas library
        data = pd.read_csv(filePath, header=None)
        return data

# global variable
problem = ''

def writetxt(filename, string):
    f = open(filename + '.txt', 'w')
    f.write(string)
    f.close()

def plot(mlp,dataframe, start, end,scale):

    global problem

    x = []
    for i in range(start*scale, end*scale):
        for j in range(start*scale,end*scale):
            x.append([i/scale,j/scale])
    x = np.array(x)
    #print(x)
    colors = []
    inputs = dataframe.drop(dataframe.columns[-1], axis=1)
    targets = dataframe.drop(dataframe.columns[:-1], axis=1).values
    for y in targets:
        colors.append('red') if y == 0 else colors.append('green')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(inputs[inputs.columns[0]], inputs[inputs.columns[1]],targets,color=colors)
    y = mlp.predict(x)
    y = [y[i][0] for i in range(len(y))]
    ax.plot_trisurf(x[:,0],x[:,1], y)
    plt.suptitle(problem)
    plt.show()

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


def _or(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    global problem
    problem = 'OR'

    # making parameters log
    writetxt('OR_Parameters',
             'Input: ' + str(inputNumber) + '\nHidden: ' + str(hiddenNumber) + '\nOutput: ' + str(
                 outputNumber) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))

    mlp = MLP(inputNumber, hiddenNumber, outputNumber)

    # making initial weights log
    writetxt('OR_Initial_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemOR.csv')
    output = mlp.train(data, epochs, learningRate)

    inputs = data.drop(data.columns[-1], axis=1).values
    for inputvalue in inputs:
        print("Input: {} Predicted Output: {} Output: {}".format(inputvalue,1 if mlp.predict([inputvalue])[0] >= 0.5 else 0, mlp.predict([inputvalue])[0]))

    # making final weights log
    writetxt('OR_Final_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemOR.csv')
    plot(mlp, data, -2, 2, 6)


def _xor(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    global problem
    problem = 'XOR'

    # making parameters log
    writetxt('XOR_Parameters',
             'Input: ' + str(inputNumber) + '\nHidden: ' + str(hiddenNumber) + '\nOutput: ' + str(
                 outputNumber) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))

    mlp = MLP(inputNumber, hiddenNumber, outputNumber)

    # making initial weights log
    writetxt('XOR_Initial_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemXOR.csv')
    output = mlp.train(data, epochs, learningRate)

    inputs = data.drop(data.columns[-1], axis=1).values
    for inputvalue in inputs:
       print("Input: {} Predicted Output: {} Output: {}".format(inputvalue,1 if mlp.predict([inputvalue])[0] >= 0.5 else 0, mlp.predict([inputvalue])[0]))

    # making final weights log
    writetxt('XOR_Final_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/problemXOR.csv')
    plot(mlp, data, -2, 2, 6)


def _characters(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    global problem
    problem = 'CHAR'

    # making parameters log
    writetxt('CHAR_Parameters',
             'Input: ' + str(inputNumber) + '\nHidden: ' + str(hiddenNumber) + '\nOutput: ' + str(
                 outputNumber) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))

    mlp = MLP(inputNumber, hiddenNumber, outputNumber)

    # making initial weights log
    writetxt('CHAR_Initial_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    data = mlp.openFile('Data/caracteres-limpo.csv')
    # mapping letters to array of values
    letters_map = {'A': np.array([1, 0, 0, 0, 0, 0, 0]),
                   'B': np.array([0, 1, 0, 0, 0, 0, 0]),
                   'C': np.array([0, 0, 1, 0, 0, 0, 0]),
                   'D': np.array([0, 0, 0, 1, 0, 0, 0]),
                   'E': np.array([0, 0, 0, 0, 1, 0, 0]),
                   'J': np.array([0, 0, 0, 0, 0, 1, 0]),
                   'K': np.array([0, 0, 0, 0, 0, 0, 1])}
    print('Testing with a clean dataframe:')
    data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: letters_map[x])

    # training and predict
    output = mlp.train(data, epochs, learningRate)
    inputs = data.drop(data.columns[-1], axis=1).values
    # making final weights log
    writetxt('CHAR_Final_Weights',
             'From input layer through hidden layer:\n' + str(mlp.weightsInputToHidden) +
             '\n\nFrom hidden layer through output layer:\n' + str(mlp.weightsHiddenToOutput))

    letters = list(letters_map.keys())

    for inputValues, outputValues, targetValues in zip(inputs, output, data[data.columns[-1]].values):
        i = max(range(len(targetValues)), key=lambda x: targetValues[x])
        print('Expected Output:', letters[i], letters_map[letters[i]], end=' ')
        i = max(range(len(outputValues)), key=lambda x: outputValues[x])
        print('Output:', letters[i], letters_map[letters[i]], outputValues)

    ''' Noisy Characters '''
    print('\nTesting with a dataframe containing messy pixels:')
    # testing on a messy pixels characters dataframe
    data = mlp.openFile('Data/caracteres-ruido.csv')
    # mapping letters to array of values
    data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: letters_map[x])

    # data treatment
    inputs = data.drop(data.columns[-1], axis=1).values
    output = mlp.predict(inputs)

    # predict
    for outputValues, targetValues in zip(output, data[data.columns[-1]].values):
        i = max(range(len(targetValues)), key=lambda x: targetValues[x])
        print('Expected Output:', letters[i], letters_map[letters[i]], end=' ')
        i = max(range(len(outputValues)), key=lambda x: outputValues[x])
        print('Output:', letters[i], letters_map[letters[i]], outputValues)


_and(2,4,1,100,0.7)
#_or(2,4,1,100,0.5)
#_xor(2,4,1,1000,0.5)
>>>>>>> 4305bff6a6e2760f32923161bb1327ec8491cb0a
#_characters(63, 20, 7, 100, 0.5)