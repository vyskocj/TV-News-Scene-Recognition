import h5py, os
import numpy as np

import keras
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten, TimeDistributed


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
UNIT_TEST = False


def create_model(input_shape, num_classes, use_lstm=True, first_part_weights=None):
    """
    Function to create Neural Network model for recognizing TV News scenes.

    :param input_shape: Tuple in format (time_steps, img_length, img_width, img_channel), where time_steps is set if and
                        only if use_lstm is True.
    :param num_classes: The number of classes to be classified by the Neural Network.
    :param use_lstm: Optional, switch that adds the ability to use LSTM network.
    :param first_part_weights: The file path of weights to already trained first part network, ie. network of this
                               function with use_lstm=False and first_part_weights=None.
    :return: The created model.
    """

    # define the number of neurons in the hidden layer of the first part model
    if UNIT_TEST:
        num_neurons_hid_layer_vgg16 = 1
    else:
        num_neurons_hid_layer_vgg16 = 1400

    # create sequential model
    model = keras.models.Sequential()

    # ==================================================================================================================
    # The first part of the Neural Network - VGG16
    # ==================================================================================================================
    # first is VGG16 without fully connected layers
    if use_lstm:
        model.add(
            TimeDistributed(
                VGG16(weights='imagenet', include_top=False),
                input_shape=input_shape
            )
        )
    else:
        model.add(
            VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        )
    # endif use_lstm // VGG16 w/o fully connected layers

    # own fully connected layers
    if use_lstm:
        model.add(TimeDistributed(Flatten()))

        model.add(TimeDistributed(Dense(num_neurons_hid_layer_vgg16, activation='relu')))
        model.add(TimeDistributed(Dropout(0.50)))

        if first_part_weights is not None:
            # complete the trained network structure to load weights
            model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
            model.load_weights(first_part_weights)

            # remove the last layer, ei. the classification layer
            model.pop()

    else:
        model.add(Flatten())

        model.add(Dense(num_neurons_hid_layer_vgg16, activation='relu'))
        model.add(Dropout(0.50))

        model.add(Dense(num_classes, activation='softmax'))

        if first_part_weights is not None:
            model.load_weights(first_part_weights)
    # endif use_lstm // own fully connected layers

    # ==================================================================================================================
    # The second part of the Neural Network - LSTM
    # ==================================================================================================================
    if use_lstm:
        model.add(LSTM(32, input_shape=(input_shape[0], num_neurons_hid_layer_vgg16)))
        model.add(Dense(num_classes, activation='softmax'))
    # endif use_lstm // the LSTM part of creating model

    return model
