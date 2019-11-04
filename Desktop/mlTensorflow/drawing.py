import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

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
    model.fit(x_train, y_train, epochs=3)
    model.save("models/test_model.h5")
    return model

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)
#print("x_test:", x_test[0])
#x_train = x_train/255.0
#y_train = y_train/255.0
new_model = GetModel()

val_loss, val_acc = new_model.evaluate(x_test, y_test)

#prediction = new_model.predict([x_test])
#singleImage = np.array(x_test)

prediction = new_model.predict([x_test])
#print("leng", len(prediction))
for i in range(5):
#    print("Prediction:", np.argmax(prediction[i]))
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title("Prediction: " + str(np.argmax(prediction[i])))
    plt.show()

#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()

