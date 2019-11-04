#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def GetModel():
    if os.path.isfile("models/test_model.h5") and not False:
        print("Model already exists")
        return keras.models.load_model("models/test_model.h5")
    print("model does not exist, creating one")

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10)
    model.save("models/test_model.h5")
    return model

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

new_model = GetModel()

val_loss, val_acc = new_model.evaluate(x_test, y_test)


"""
cImage = keras.preprocessing.image.load_img("2.jpg")

ArrayImage = keras.preprocessing.image.img_to_array(cImage)
print(ArrayImage)

x_Test_Image = keras.utils.normalize(ArrayImage, axis=1)

print(x_Test_Image.shape)


singleImage = x_test[0].reshape(1, 28, 28)

"""

img_arr = cv2.imread("Big_6.png", cv2.IMREAD_GRAYSCALE)

x_Test_Image = keras.utils.normalize(img_arr, axis=1)
#plt.grid(False)
#plt.imshow(x_Test_Image, cmap=plt.cm.binary)
#plt.show()
test = cv2.resize(img_arr, (28, 28))
singleImage = test.reshape(-1, 28, 28)
#print(singleImage.shape)


prediction = new_model.predict(singleImage)

plt.grid(False)
plt.imshow(singleImage[0], cmap=plt.cm.binary)
plt.title("Prediction: " + str(np.argmax(prediction)))
plt.show()

