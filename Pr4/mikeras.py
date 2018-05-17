import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import rmsprop
from sklearn.datasets import fetch_mldata
"""
mnist = fetch_mldata('MNIST original', data_home='~/Documents/Universidad/GCOM/Pr4')
data = mnist.data
target = mnist.target
mis_digitos = np.hstack([data, target.reshape(data.shape[0], 1)])
np.random.shuffle(mis_digitos)
ochenta = int(np.rint(mis_digitos.shape[0] * 0.8))
train = mis_digitos[:ochenta, :]
train_x = train[:, :train.shape[1] - 1]
train_t = train[:, train.shape[1] - 1:]
test = mis_digitos[ochenta:, :]
test_x = test[:, :test.shape[1] - 1]
test_t = test[:, test.shape[1] - 1:]

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1) # anadir que tiene un unico canal (no es RGB) y normalizar a [0,1]
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x /= 255
test_x /= 255

train_t = np_utils.to_categorical(train_t, 10) # Convertir en vectores canonicos
test_t = np_utils.to_categorical(test_t, 10)

model = Sequential()  # Crear red convolucional
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Preparar su funcion de coste y ?optimizador?

model.fit(train_x, train_t, batch_size=32, nb_epoch=3, verbose=1)  # Entrenar la red

score = model.evaluate(test_x, test_t, verbose=0)
"""
from keras.datasets import cifar10

(train_x, train_t), (test_x, test_t) = cifar10.load_data()

#train_x = np.array(map(lambda x: np.transpose(x), train_x)) # anadir que tiene 3 canales y es
#test_x = np.array(map(lambda x: np.transpose(x), test_x))
train_x = train_x.astype('float32')
train_t = train_t.astype('float32')
train_x /= 255
train_t /= 255

train_t = np_utils.to_categorical(train_t, 10) # Convertir en vectores canonicos
test_t = np_utils.to_categorical(test_t, 10)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=train_x.shape[1:]))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # Preparar su funcion de coste y ?optimizador?

model.fit(train_x, train_t, batch_size=32, nb_epoch=3, verbose=1, validation_data=(test_x, test_t),  shuffle=True)  # Entrenar la red

score = model.evaluate(test_x, test_t, verbose=0)

print(score)