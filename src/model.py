import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def train_model(file):
    minst = tf.keras.datasets.mnist

    (X_train, y_train),(X_test, y_test)= minst.load_data()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=4)
    img = np.invert(np.array([file]))
    return np.argmax(model.predict(img))

