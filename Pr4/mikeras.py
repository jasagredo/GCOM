from __future__ import print_function
import keras
from sklearn.datasets import fetch_mldata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = 10
epochs = 12

mnist = fetch_mldata('MNIST original', data_home='~/Documents/Universidad/GCOM/Pr4')
data = mnist.data
target = mnist.target
mis_digitos = np.hstack([data, target.reshape(data.shape[0], 1)])
np.random.shuffle(mis_digitos)
ochenta = int(np.rint(mis_digitos.shape[0] * 0.8))
train = mis_digitos[:ochenta, :]
train_x = train[:, :train.shape[1] - 1].T
train_t = train[:, train.shape[1] - 1:]
test = mis_digitos[ochenta:, :]
test_x = test[:, :test.shape[1] - 1].T
test_t = test[:, test.shape[1] - 1:]

train_x /= 255
test_x /= 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=((1, 784, 1))))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=mikeras.losses.categorical_crossentropy,
              optimizer=mikeras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_x, train_t,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_x, test_t))
score = model.evaluate(test_x, test_t)
print('Test loss:', score[0])
print('Test accuracy:', score[1])