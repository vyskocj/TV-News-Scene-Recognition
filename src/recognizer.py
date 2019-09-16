import h5py
import os
import xlwt
import matplotlib.pyplot as plt
from datetime import datetime

import keras
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, TimeDistributed
from keras.layers import Dense, Dropout, Flatten

from src.const_spec import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
UNIT_TEST = False


def create_model(input_shape, num_classes, use_lstm=True, first_part_trainable=True, first_part_weights=None):
    """
    Function to create Neural Network model for recognizing TV News scenes.

    :param input_shape: Tuple in format (num_frames, frame_height, frame_width, frame_channels), where num_frames is set
                        if and only if use_lstm is True.
    :param num_classes: The number of classes to be classified by the Neural Network.
    :param use_lstm: Optional, switch that adds the ability to use LSTM network.
    :param first_part_trainable: Optional, by switching off is not the first part of model trainable. It can be used
                                 if use_lstm is True.
    :param first_part_weights: Optional, the file path of weights to already trained first part network, ie. network of
                               this function with use_lstm=False and first_part_weights=None.
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
                input_shape=input_shape,
                trainable=first_part_trainable
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

        model.add(
            TimeDistributed(
                Dense(num_neurons_hid_layer_vgg16, activation='relu'),
                trainable=first_part_trainable
            )
        )
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


def train_model(model, train_data, validation_data, epochs, batch_size, optimizer, output_path=None):
    """
    Train model with the option to generate progress.

    :param model: Instance to the Neural Network to be trained.
    :param train_data: Tuple in format (data, labels). The data shape must be in input_shape format extended by
                       batch_shape, i.e. number of training data, the labels shape must be extended too.
    :param validation_data: Same as train_data.
    :param epochs: The number of epochs for the model to be trained.
    :param batch_size: The number of samples that will be propagated through the network.
    :param optimizer: Keras optimizer instance.
    :param output_path: Optional, path where is generated training history and where is saved trained model.
    :return: History of training and output path.
    """
    # set the output directory name
    if output_path is not None:
        if not os.path.exists(output_path):
            print('[E] Output path is invalid!\n'
                  '- You are currently: ' + os.path.abspath('.') + '\n'
                  '- Your request was: ' + output_path + '\n')
            return INVALID_OUTPUT_PATH

        output_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        output_dir = ''  # [W]: Local variable 'output_dir' might be referenced before assignment
    # if output_path is not None // set the output directory name

    # ==================================================================================================================
    # Part of training the model
    # ==================================================================================================================
    # setting the compiler for training and fitting the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(train_data[0], train_data[1], validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs, shuffle='batch', verbose=(0 if UNIT_TEST else 1))

    # ==================================================================================================================
    # Part of the output generation
    # ==================================================================================================================
    # generate an output if required
    if output_path is not None:
        # making a new directory by the date time
        output = output_path + ('\\' if output_path[-1] != '\\' else '') + output_dir + '\\'
        os.mkdir(output)

        # preparing an Excel file for saving the training progress
        book = xlwt.Workbook(encoding='utf-8')

        # saving the output to the new directory
        for cat in ['acc', 'loss']:
            # ==========================================================================================================
            # Plotting the graphs
            # ==========================================================================================================
            plt.figure(figsize=(20, 15))

            plt.plot(history.history[cat])
            plt.plot(history.history['val_' + cat])

            plt.title('model ' + cat)
            plt.ylabel(cat)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            plt.savefig(output + cat + '.svg')
            plt.clf()

            # ==========================================================================================================
            # Saving the numeric data
            # ==========================================================================================================
            sheet = book.add_sheet(cat)
            sheet.write(0, 0, 'Epoch')
            sheet.write(0, 1, cat)
            sheet.write(0, 2, 'Validation ' + cat)

            # saving the training history
            for e in range(1, epochs + 1):
                sheet.write(e, 0, e - 1)
                sheet.write(e, 1, history.history[cat][e - 1])
                sheet.write(e, 2, history.history['val_' + cat][e - 1])
            # endfor e in range(1, epochs + 1) // saving the training history
        # endfor type in ['acc', 'loss'] // saving the output to the new directory

        # save the Excel file
        book.save(output + 'history.xls')

        # ==============================================================================================================
        # Saving the model summary
        # ==============================================================================================================
        with open(output + 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=120)

        # ==============================================================================================================
        # Saving the model
        # ==============================================================================================================
        model.save(output + 'model.h5')

        if not UNIT_TEST:
            print('\n\n')
            print(f'The output was successfully saved: %s' % output)
    # endif output_path is not None // generate the output if required

    return history, output
