# =================================================================================
# PROBLEM A1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1].
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-4
# =================================================================================


import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<1e-4):
            print("\nSudah Mencapai Desired loss (MSE) < 1e-4")
            self.model.stop_training = True


def solution_A1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)

    # YOUR CODE HERE
    model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
    ])

    epoch=1000
    callback = MyCallback()
    model.compile(optimizer='sgd', loss='mse')
    model.fit(X, Y, epochs=epoch, callbacks=[callback])  # You may need to adjust the number of epochs for convergence
    pred_X = np.array([-2.0, 10.0]).reshape([-1, 1])
    predicted_Y = model.predict(pred_X)
    print(predicted_Y)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A1()
    model.save("model_A1.h5")