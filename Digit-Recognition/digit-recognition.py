###############################################################################
# Filename: digit-recognition
#
# Description: This script builds a simple neural network for digit recognition
#              following a tutorial by NeuralNine on YouTube (Reference)
#
# Inputs (lines 32-38):
#   TRAIN_MODEL:    boolean indicating if a new model should be trained
#   image_size:     size of input images
#   hl_node_count:  hidden layer node count
#   hl_activ_func:  hidden layer activation function
#   optimizer_func: optimizer function for training
#   loss_func:      loss function
#   mdl_name:       trained model name
#
# Outputs: handwritten_digits.keras
#
# References:
#   Neural Network Python Project - Handwritten Digit Recognition
#   (https://www.youtube.com/watch?v=bte8Er0QhDg)
#
###############################################################################
from supportFuncs import *   # for image testing function
from pathlib import Path     # for saving trained model
import tensorflow as tf      # for machine learning

## USER INPUTS ##
TRAIN_MODEL = False   # toggle to train & test OR just test
image_size = (28, 28)
hl_node_count = 128
hl_activ_func = 'relu'
optimizer_func = 'adam'
loss_func = 'sparse_categorical_crossentropy'
mdl_name = 'handwritten_digits.keras'

## NEURAL NETWORK CODE ##
if TRAIN_MODEL:

    # get the training and testing data
    mnist = tf.keras.datasets.mnist
    (writDig_train, classDig_train), (writDig_test, classDig_test) = mnist.load_data()

    # normalize the digit pixel data [0, 255] --> [0, 1]
    writDig_train = tf.keras.utils.normalize(writDig_train, axis=1)
    writDig_test = tf.keras.utils.normalize(writDig_test, axis=1)

    # create the neural network (Sequential model)
    model = tf.keras.models.Sequential()

    # add layers to the neural network
    model.add(tf.keras.layers.Flatten(input_shape=image_size))                  # 1. input image size (flattened)
    model.add(tf.keras.layers.Dense(hl_node_count, activation=hl_activ_func))   # 2. dense layer with relu activation
    model.add(tf.keras.layers.Dense(hl_node_count, activation=hl_activ_func))   # 3. dense layer with relu activation
    model.add(tf.keras.layers.Dense(10, activation='softmax'))                  # 4. dense layer for digit outputs (0-9) with softmax activation

    # compile the neural network (prepare the training optimizer, loss function, and data to log)
    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=['accuracy'])

    # train the model over some epochs and save
    model.fit(writDig_train, classDig_train, epochs=3)
    model.save(Path(__file__).resolve().parent / mdl_name)

    # load the trained model and evaulate
    model = tf.keras.models.load_model(Path(__file__).resolve().parent / mdl_name)
    loss, accuracy = model.evaluate(writDig_test, classDig_test)
    print(f'\n=== Handwritten Digits Identifier Model ===\nLoss: {loss}\nAccuracy: {accuracy}\n')

# test the trained model on custom images
drawAndPredictDigit(model_path=Path(__file__).resolve().parent / mdl_name)
