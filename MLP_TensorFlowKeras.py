# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Bibliotecas auxiliares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print(tf.__version__)

# Lê um arquivo txt e retorna o seu conteúdo (data)
def openFile(filePath):

    # função usada para abrir os arquivos contendo os dados de treinamento e teste
    data = pd.read_csv(filePath, header=None)
    return data

# Cria um arquivo filename.txt, onde filename é um parâmetro passado na chamada da função. Escreve nesse arquivo uma string passada na implementação do código
def writetxt(filename, string):
    f = open(filename + '.txt', 'w')
    f.write(string)
    f.close()

# retorna qual à qual letra se refere a combinação de números 
def getLetterByVector(dictOfElements, valueToFind): 
    for letter, combination in dictOfElements.items():
        if (combination == valueToFind).all(): 
            return letter
    else:
        return -1

# Estrutura de chamada para cada caso de teste
#AND
def _and(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    default_cases('AND', inputNumber, hiddenNumber, outputNumber, epochs, learningRate)

#XOR
def _xor(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    default_cases('XOR', inputNumber, hiddenNumber, outputNumber, epochs, learningRate)

#OR
def _or(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    default_cases('OR', inputNumber, hiddenNumber, outputNumber, epochs, learningRate)

#CHAR
def _chars(inputNumber, hiddenNumber, outputNumber, epochs, learningRate):
    default_cases('caracteres', inputNumber, hiddenNumber, outputNumber, epochs, learningRate)

# Estrutura essencial para todos os casos de teste, nela estão contidas a criação da rede e dos arquivos de saída
def default_cases(problem_choosed, input_number, hidden_number, output_number, epochs, learningRate):
    global problem
    
    # Criando o arquivo de log dos parâmetros da rede
    writetxt(problem_choosed + '_Parameters''Input: ' + str(input_number) + '\nHidden: ' + str(hidden_number) + '\nOutput: ' + str(
                 output_number) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))
    
    # criando a estrutura da rede neural
    # o "modelo" da rede está especificado dentro de "model". Pode-se ver que criamos uma estrutura sequencial com todas as layers da MLP. A estrutura é Sequencial é mais simples que a funcioal do Keras,
    # porém é suficiente para realizar a interação(e atualização de pesos) entre os neurônios de uma camada e outra.
    # keras.layers.Dense é responsável por criar uma layer no modelo, conectando todos neurônios de saída com todos de entrada da próxima camada, sendo portanto, densa
    # dentro de todos seus parâmetros, utilizamos o número de saídas da camada (primeiro parâmetro), o formato da entrada (segundo parâmetro) e especificamos a função de ativação no terceiro parâmetro
    model = keras.Sequential([ # Definição da estrutura do modelo como Sequencial
        keras.layers.Dense(hidden_number, input_shape = (input_number,), activation='sigmoid'), # Criação dsc
        keras.layers.Dense(output_number, activation='sigmoid')
        
        #estão implícitos os seguintes parâmetros:
        #use_bias=True,
        #kernel_initializer="glorot_uniform",
        #bias_initializer="zeros",
        #kernel_regularizer=None,
        #bias_regularizer=None,
        #activity_regularizer=None,
        #kernel_constraint=None,
        #bias_constraint=None,
    ]) 

    # setting the methods that will be used on the NNno
    #Construir a rede neural requer configurar as camadas do modelo, e depois, compilar o modelo.
    
    #Compilar o Modelo usando Sequential Model
    #O modelo Sequencial, que é muito direto (uma lista simples de camadas), mas está limitado a pilhas de camadas de entrada única e saída única (como o nome indica).
    #Função Loss —Essa mede quão preciso o modelo é durante o treinamento. Queremos minimizar a função para guiar o modelo para direção certa.
    #Optimizer - Como o modelo se atualiza com base no dado que ele vê e sua função loss.
    #Métricas —usadas para monitorar os passos de treinamento e teste. O exemplo abaixo usa a acurácia, a fração das imagens que foram classificadas corretamente.
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learningRate), #Usando o metodo gradiente-descentente para otimizar os pesos
        loss='mse', # Metodo de erro usado para propagar os erros: erro quadratico médio
        metrics=['accuracy']) # Metodos para calcular os erros
    
    weights = model.get_weights()
    
    # fazendo log de pesos iniciais
    writetxt(problem_choosed + '_Initial_Weights',
             'From input layer through hidden layer:\n' + str(weights[0]) + '\nBias:\n' + str(weights[1]) +
             '\n\nFrom hidden layer through output layer:\n' + str(weights[2]) + '\nBias:\n' + str(weights[3]))
    
    # obtendo os dados do csv
    if problem_choosed != 'caracteres':
        data = openFile('Data/problem'+ problem_choosed +'.csv')
    else :
        data = openFile('Data/caracteres-limpo.csv')
        test_data = openFile('Data/caracteres-ruido.csv')
        test_inputs = inputs = data.drop(data.columns[-1], axis=1).values

    # ajustando entradas e metas
    inputs = data.drop(data.columns[-1], axis=1).values

    if problem_choosed != 'caracteres' :
        targets = data.drop(data.columns[:-1], axis=1).values
    else:
        targets = np.array([[1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1]])


    # treinando o modelo
    '''
    model.fit argumentos
    
    object : the model to train.      
    -> X : our training data. Can be Vector, array or matrix      
    -> Y : our training labels. Can be Vector, array or matrix       
    -> Batch_size : it can take any integer value or NULL and by default, it will
    be set to 32. It specifies no. of samples per gradient.      
    -> Epochs : an integer and number of epochs we want to train our model for.      
    -> Verbose : specifies verbosity mode(0 = silent, 1= progress bar, 2 = one
    line per epoch).      
    -> Shuffle : whether we want to shuffle our training data before each epoch.      
    -> steps_per_epoch : it specifies the total number of steps taken before
    one epoch has finished and started the next epoch. By default it values is set to NULL.
    '''
    hist = model.fit(inputs, targets, epochs=epochs)

    # produzindo o log contendo o erro por epoca atraves do historico armazenado na variavel hist
    errors = ''
    for i in range(len(hist.history['loss'])):
        errors += 'Epoch ' + str(i) + ': Error = ' + str(hist.history['loss'][i]) + '\n'
    writetxt(problem_choosed + '_Errors_Per_Epoch', errors)

    # criando o arquivo de log dos pesos
    weights = model.get_weights()
    writetxt(problem_choosed + '_Final_Weights',
             'From input layer through hidden layer:\n' + str(weights[0]) + '\nBias:\n' + str(weights[1]) +
             '\n\nFrom hidden layer through output layer:\n' + str(weights[2]) + '\nBias:\n' + str(weights[3]))

    
    # criando o arquivo de log da predição
    prediction = 'General Accuracy: {}\n'.format(hist.history['accuracy'][i])
    if problem_choosed != 'caracteres' :
        predicted = model.predict(inputs)
        for input, target, predict in zip(inputs,targets, predicted):
            prediction += "Input: {} Expected Output: {} Predicted Output: {} Predicted Output Int: {}\n".format(input, target, predict, 1 if predict[0] >= 0.5 else 0)
    else:
        predicted = model.predict(test_inputs)
        
        letters_map = {
            'A': np.array([1, 0, 0, 0, 0, 0, 0]),
            'B': np.array([0, 1, 0, 0, 0, 0, 0]),
            'C': np.array([0, 0, 1, 0, 0, 0, 0]),
            'D': np.array([0, 0, 0, 1, 0, 0, 0]),
            'E': np.array([0, 0, 0, 0, 1, 0, 0]),
            'J': np.array([0, 0, 0, 0, 0, 1, 0]),
            'K': np.array([0, 0, 0, 0, 0, 0, 1])}    
        
        
        for input, target, predict in zip(test_inputs,targets, predicted):
            predictLetterInt = [(1 if i >= 0.5 else 0) for i in predict]
            prediction += "Input: {} Expected Output: {} Predicted Output Int: {} Expected Output Letter: {} Predicted Output Letter: {} Predicted Output: {}\n".format(input, target, predictLetterInt, getLetterByVector(letters_map, target), getLetterByVector(letters_map, predictLetterInt), predict)

    writetxt( problem_choosed + '_Outputs', prediction)




#_and(2,4,1,100,0.5)
#_or(2,4,1,200,0.5)
#_xor(2,4,1,500,0.5)
_chars(63,20,7,500,0.5)

