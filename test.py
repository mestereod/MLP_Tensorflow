import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
import pydot

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]])

# the four expected results in the same order
target_data = np.array([[0],[0],[0],[1]])

model = Sequential()
xa = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
xb = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
l1 = Dense(4, input_dim=2, activation='sigmoid', use_bias=True, weights = [2*np.random.rand(2,4)-1,2*np.random.rand(4)-1])
model.add(l1)
model.add(Dense(1, activation='sigmoid', use_bias = True, weights = [2*np.random.rand(4,1)-1,2*np.random.rand(1)-1]))

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.75, name="SGD"),
              metrics=['accuracy'])

print(model.get_weights())

model.fit(training_data, target_data, nb_epoch=100, verbose=2)

print ('{} \n {}'.format(model.predict(training_data),model.predict(training_data).round()))

print(model.get_weights())

''' Weights with bias
[
 array([[ 0.31961277, -0.15061027,  0.77195275,  0.51203406],
       [ 0.7831835 ,  0.8858454 ,  0.4992858 , -0.01089129]],
      dtype=float32),
 array([1.4028091, 0.845695 , 1.1394044, 1.3944246], dtype=float32), 
 array([[-0.8417013 ],
       [ 0.5475723 ],
       [-0.17419438],
       [-1.0409423 ]], dtype=float32), 
 array([0.25507814], dtype=float32)
]
'''

''' Weights with no bias
[
    array([[ 0.5848363 , -0.27213407,  0.15466999, -0.7349917 ],
           [ 0.7776543 , -0.4604378 ,  1.0661523 , -1.2363087 ]],dtype=float32), 
    array([[ 0.17650414],
            [-1.172227  ],
            [ 0.314953  ],
            [-2.3453846 ]], dtype=float32)
]
'''