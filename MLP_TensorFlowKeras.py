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

# Retorna qual à qual letra se refere a combinação de números 
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

    # Criando o arquivo de log dos parâmetros da rede
    writetxt(problem_choosed + '_Parameters',
             'Input: ' + str(input_number) + '\nHidden: ' + str(hidden_number) + '\nOutput: ' + str(
                 output_number) + '\nEpochs: ' +
             str(epochs) + '\nLearning Rate: ' + str(learningRate))
    
    # criando a estrutura da rede neural
    # o "modelo" da rede está especificado dentro de "model". Pode-se ver que criamos uma estrutura sequencial com todas as layers da MLP. A estrutura é Sequencial é mais simples que a funcioal do Keras,
    # porém é suficiente para realizar a interação(e atualização de pesos) entre os neurônios de uma camada e outra.
    # keras.layers.Dense é responsável por criar uma layer no modelo, conectando todos neurônios de saída com todos de entrada da próxima camada, sendo portanto, densa
    # dentro de todos seus parâmetros, utilizamos o número de saídas da camada (primeiro parâmetro), o formato da entrada (segundo parâmetro) e especificamos a função de ativação no terceiro parâmetro
    model = keras.Sequential([ 
        keras.layers.Dense(hidden_number, input_shape = (input_number,), activation='sigmoid'), 
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

    #Construir a rede neural requer configurar as camadas do modelo, e depois, compilar o modelo.

    # Compilar o Modelo usando Sequential Model
    # O modelo Sequencial, que é muito direto (uma lista simples de camadas), mas está limitado a pilhas de camadas de entrada única e saída única (como o nome indica).
    # Função Loss —Essa mede quão preciso o modelo é durante o treinamento. Queremos minimizar a função para guiar o modelo para direção certa.
    # Optimizer - É como o modelo se atualiza com base no dado que ele vê e sua função loss.
    # Métricas —usadas para monitorar os passos de treinamento e teste. O exemplo abaixo usa a acurácia, a fração das imagens que foram classificadas corretamente.
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learningRate), #Usando o metodo gradiente-descentente para otimizar os pesos
        loss='mse', # Metodo de erro usado para propagar os erros: erro quadratico médio
        metrics=['accuracy']) # Metodos para calcular os erros
    
    # Armazenamos os pesos gerados pós compilarmos o modelo em um arranjo e, como esses valores mudarão quando treinarmos o modelo, já guardamos seus valores
    weights = model.get_weights()
    
    # fazendo log de pesos iniciais
    # Utilizamos os pesos da variável weights, com os pesos dos neurônios e do Bias
    weights_concatenate = ''
    for i in weights[0]:
        weights_concatenate += str(i) + " "
    writetxt(problem_choosed + '_Initial_Weights',
             'From input layer through hidden layer:\n' + weights_concatenate + '\nBias:\n' + str(weights[1]) +
             '\n\nFrom hidden layer through output layer:\n' + str(weights[2]) + '\nBias:\n' + str(weights[3]))
    
    # obtendo os dados do csv
    # Caso seja escolhido o método AND, OR ou XOR, chama a função openFile, fornecendo o caminho para o arquivo
    if problem_choosed != 'caracteres':
        data = openFile('Data/problem'+ problem_choosed +'.csv')
    # Caso seja o problema dos caracteres, abre tanto o arquivo com os carateres limpos quanto o ruído e 
    # armazena seus conteúdos (exceto a última coluna, contendo a letra) em duas matrizes, cada uma referente ao tipo de arquivo que leu
    else :
        data = openFile('Data/caracteres-limpo.csv')
        test_data = openFile('Data/caracteres-ruido.csv')
        inputs = data.drop(data.columns[-1], axis=1).values
        test_inputs = test_data.drop(test_data.columns[-1], axis=1).values

    # ajustando entradas e metas
    # a entrada será os dados contidos no arquivo csv, exceto a última coluna (label)
    inputs = data.drop(data.columns[-1], axis=1).values
    # Se for o problema AND, OR ou XOR o target (resultado esperado) serão os dados contidos na última coluna (label)
    if problem_choosed != 'caracteres' :
        targets = data.drop(data.columns[:-1], axis=1).values
    else: 
    # Como existem três repetições para cada caractere, colocamos três repetições para as representações de cada letra
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


    # Treinamos o modelo utilizando o método fit
    # Passamos nesse método os valores de entrada que e os valores que esperamos para saída, ambos lidos do CSV anteriormente
    # Adicionamos a quantidade de épocas também, e tudo isso fica na variável hist
    hist = model.fit(inputs, targets, epochs=epochs)

    # produzindo o log contendo o erro por epoca atraves do historico armazenado na variavel hist
    # Para isso, inicializamos uma string errors, essa string será o conteúdo do txt
    # Para cada reconhecimento do campo loss (Erro por época) que temos no nosso modelo, pegamos ele adicionamos na nossa string,
    # apontando o erro na época, em todas as épocas (pois o campo loss existe para todas as épocas).
    errors = ''
    for i in range(len(hist.history['loss'])):
        errors += 'Epoch ' + str(i) + ': Error = ' + str(hist.history['loss'][i]) + '\n'
    writetxt(problem_choosed + '_Errors_Per_Epoch', errors)

    # criando o arquivo de log dos pesos
    # após treinar o modelo, armazenamos em weights os novos pesos
    # esses novos pesos são os pesos finais que serão utilizados no log Final Weights
    weights = model.get_weights()
    
    weights_concatenate = ''
    for i in weights[0]:
        weights_concatenate += str(i) + " "
        
    writetxt(problem_choosed + '_Final_Weights',
             'From input layer through hidden layer:\n' + weights_concatenate + '\nBias:\n' + str(weights[1]) +
             '\n\nFrom hidden layer through output layer:\n' + str(weights[2]) + '\nBias:\n' + str(weights[3]))

    
    # criando o arquivo de log da predição / outputs
    prediction = ''
    
    # Se o problema for AND, OR ou XOR
    if problem_choosed != 'caracteres' :
        # Cada valor predito será calculado pelo nosso modelo baseado em um par de valores de entrada (por exemplo, 1 e 1).
        # Os valores preditos serão um valor de 0 à 1.
        predicted = model.predict(inputs)
        # Monta-se tuplas baseadas nos valores de entrada, saída esperada e valores de saída calculados (preditos) para serem usados no log
        for input, target, predict in zip(inputs,targets, predicted):
            # Imprime os três conjuntos de valores citados anteriormente e um novo, chamado Predicted Output Int, que arredonda o valor predito para 1 caso seu valor
            # seja maior ou igual à 0.5, e caso seja menor arredondará para 0 
            prediction += "Input: {} Expected Output: {} Predicted Output: {} Predicted Output Int: {}\n".format(input, target, predict, 1 if predict[0] >= 0.5 else 0)
    
    # Se for o problema dos caracteres 
    else:
        # Utiliza os inputs do arquivo csv caracteres-limpo para calcular os valores preditos
        predicted = model.predict(test_inputs)
        
        # O mapa mostra qual letra é representada por qual sequência de números
        letters_map = {
            'A': np.array([1, 0, 0, 0, 0, 0, 0]),
            'B': np.array([0, 1, 0, 0, 0, 0, 0]),
            'C': np.array([0, 0, 1, 0, 0, 0, 0]),
            'D': np.array([0, 0, 0, 1, 0, 0, 0]),
            'E': np.array([0, 0, 0, 0, 1, 0, 0]),
            'J': np.array([0, 0, 0, 0, 0, 1, 0]),
            'K': np.array([0, 0, 0, 0, 0, 0, 1])}    
        
        # Monta-se tuplas baseadas nos valores de testes, saída esperada e valores de saída calculados (preditos) para serem usados no log
        for input, target, predict in zip(test_inputs, targets, predicted):
            # A variável predictLetterInt é 1 quando o valor em cada índice predito no arrranjo é maior ou igual à 0.5, se for menor arredondamos para 0.
            predictLetterInt = [(1 if i >= 0.5 else 0) for i in predict]
            ''' 
            Para cada um dos valore a ser predito, escreve no log :
            A entrada do CSV,
            A saída esperadOa no nosso formato,
            O output esperado no nosso formato,
            A letra esperada de saída, utilizando a função getLetterByVector para identificar qual letra em letters_map tem a sequência de valores representada por target
            A letra que foi predita, utilizando também a função getLetterByVector, agora identificando se letters_map tem a sequência de valores calculada à pouco em predictLetterInt
            O valor predito em nossa representação, sem os tratamentos para identificar a letra.
            '''
            prediction += "Input: {} Expected Output: {} Predicted Output Int: {} Expected Output Letter: {} Predicted Output Letter: {} Predicted Output: {}\n".format(input, target, predictLetterInt, getLetterByVector(letters_map, target), getLetterByVector(letters_map, predictLetterInt), predict)

    # Escreve o log de Output para cada problema
    writetxt( problem_choosed + '_Outputs', prediction)

#EXECUÇÃO DOS MÉTODOS
#Parâmetros (em ordem): Número de neurônios da camada de entrada, número de neurônios da camada escondida, número de neurônios de saída, quantidade de épocas e taxa de aprendizado 

#_and(2,4,1,100,0.5)
#_or(2,4,1,200,0.5)
#_xor(2,4,1,500,0.5)
_chars(63,20,7,500,0.5)
