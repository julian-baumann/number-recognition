#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pygame, sys

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

inputValue = input("Number u goin to draw\n")
lastXPos = None
lastYPos = None

def GetModel():
    if os.path.isfile("models/test_model.h5"):
        print("Model already exists")
        return keras.models.load_model("models/test_model.h5")
    print("model does not exist, creating one")

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5)
    model.save("models/test_model.h5")
    return model


def Predict():
    pygame.image.save(screen, inputValue + ".png")
    new_model = GetModel()

    val_loss, val_acc = new_model.evaluate(x_test, y_test)


    img_arr = cv2.imread(inputValue + ".png", cv2.IMREAD_GRAYSCALE)

    x_Test_Image = keras.utils.normalize(img_arr, axis=1)
    test = cv2.resize(img_arr, (28, 28))
    singleImage = test.reshape(-1, 28, 28)

    prediction = new_model.predict(singleImage)

    plt.grid(False)
    plt.imshow(singleImage[0], cmap=plt.cm.binary)
    plt.title("Prediction: " + str(np.argmax(prediction)))
    plt.show()

def Draw():
    mouseX, mouseY = pygame.mouse.get_pos()
    pygame.draw.circle(screen, [255, 255, 255], (mouseX, mouseY), 30);
    pygame.display.flip()

screen = pygame.display.set_mode([800, 800])
screen.fill([0, 0, 0])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                Predict()

    MouseClick = pygame.mouse.get_pressed()

    if MouseClick == (1, 0, 0):
        Draw()







