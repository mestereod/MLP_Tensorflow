import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
import pydot

# the four different states of the XOR gate
training_data = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[0],[0],[1]], "float32")

model = Sequential()
l1 = Dense(4, input_dim=2, activation='sigmoid', use_bias=True, bias_initializer='ones')
model.add(l1)
model.add(Dense(1, activation='sigmoid', use_bias = True, bias_initializer = 'ones'))

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.5, name="SGD"),
              metrics=['accuracy'])

model.fit(training_data, target_data, nb_epoch=400, verbose=2)

print (model.predict(training_data).round())


