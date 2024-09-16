import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train , y_train) , (x_test , y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis= 1)
x_test = tf.keras.utils.normalize(x_test, axis= 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

loss, accuracy = model.evaluate(x_test, y_test)
print("loss=" ,loss)
print("accuracy=",(accuracy*100),"%")

