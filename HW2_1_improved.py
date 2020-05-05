import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.layers import Dropout
import time

# data import
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 스케일 변환
x_train = x_train.reshape(60000, 784)
x_train = x_train/255
x_test = x_test.reshape(10000, 784)
x_test = x_test/255
# one-hot 벡터로 변환
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

# 중간층
model.add(
    Dense(
        units=128,
        input_shape=(784,),
        activation='relu'
    )
)
model.add(Dropout(0.2))
model.add(
    Dense(
        units=64,
        activation='relu'
    )
)

#출력층
model.add(
    Dense(
        units=10,
        activation='softmax'
    )
)
startTime = time.time()
#학습
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb=TensorBoard(log_dir='.\logs\hw1_i',
                write_graph=True,
                write_images=True,)
history_adam=model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.2,
    callbacks=[tsb]
)
endTime=time.time() - startTime
print(endTime)

plot_model(model, to_file='model1_I.png')