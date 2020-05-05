import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import Conv2D, Flatten, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, BatchNormalization, Dense
import numpy as np
import time

# data import
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 스케일 변환
print(x_train.shape)
x_train = x_train/255
x_test = x_test/255
x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))
# one-hot 벡터로 변환
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

# layer1 - conv
model.add(
    Conv2D(
        filters=20, input_shape=(28, 28, 1),
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# layer2 - conv
model.add(
    Conv2D(
        filters=40,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Fully connection layer
# ----------------------------------------
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

startTime = time.time()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb=TensorBoard(log_dir='.\logs\hw2',
                write_graph=True,
                write_images=True,)
# start study
history_model1 = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.2,
    callbacks=[tsb]
)

endTime=time.time() - startTime
print(endTime)

plot_model(model, to_file='model2.png')