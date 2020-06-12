# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librariesauxiliares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print(tf.__version__)

#Carregando os dados de treinamento
data = pd.read_csv('Data/problemAND.csv', header=None)
input_table_train =  data.drop(data.columns[-1], axis=1).values
labels = data.drop(data.columns[:-1], axis=1).values
print(input_table_train)
print(labels)

#Carregando os dados de teste
#input_table_test = 
#test_labels =

#Construindo o modelo
#Construir a rede neural requer configurar as camadas do modelo, e depois, compilar o modelo.

model = keras.Sequential([
    
    
    #keras.layers.Dense(
    #units,
    #activation=None,
    #use_bias=True,
    #kernel_initializer="glorot_uniform",
    #bias_initializer="zeros",
    #kernel_regularizer=None,
    #bias_regularizer=None,
    #activity_regularizer=None,
    #kernel_constraint=None,
    #bias_constraint=None)
    keras.layers.Dense(4, input_dim = 2, activation='sigmoid'),
    keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
]) 

#Tipos de Modelo
#The Sequential model, which is very straightforward (a simple list of layers), but is limited to single-input, single-output stacks of layers (as the name gives away).
#The Functional API, which is an easy-to-use, fully-featured API that supports arbitrary model architectures. For most people and most use cases, this is what you should be using. This is the Keras "industry strength" model.
#Model subclassing, where you implement everything from scratch on your own. Use this if you have complex, out-of-the-box research use cases.

#Compilar o Modelo usando Sequential Model
#Função Loss —Essa mede quão preciso o modelo é durante o treinamento. Queremos minimizar a função para guiar o modelo para direção certa.
#Optimizer —Isso é como o modelo se atualiza com base no dado que ele vê e sua função loss.
#Métricas —usadas para monitorar os passos de treinamento e teste. O exemplo abaixo usa a acurácia, a fração das imagens que foram classificadas corretamente.

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['accuracy'])

#Treinar o modelo
#O modelo aprende como associar as imagens as labels.

print(input_table_train)
print(labels)

model.fit(input_table_train, labels, epochs=100)

#Avalie a acurácia
test_loss, test_acc = model.evaluate(input_table_train,  labels, verbose=2)

print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc)

#Predições
predictions = model.predict(input_table_train)

#https://keras.io/api