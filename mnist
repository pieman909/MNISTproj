import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)

model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer= "adam",
		     loss= "sparse_categorical_crossentropy",
		     metrics = ["accuracy"])
model.fit(x_train, y_train, epochs=8)
model.fit(x_test, y_test, epochs=8)
valLoss, valAcc = model.evaluate(x_test, y_test)

print(valLoss, valAcc)

plt.title(y_train[2])
plt.imshow(x_train[2])
