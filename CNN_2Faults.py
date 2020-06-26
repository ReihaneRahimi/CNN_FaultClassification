from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
from numpy import load
import pickle

NAME = "CNN_Classification_2Faults"
# tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

#to load the data
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
Y = pickle.load(pickle_in)
Y = np.array (Y)

input_shape=(100,100,1)

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y,
          batch_size=32,
          epochs=20,
          validation_split=0.3)