import os
import xlwt
import matplotlib.pyplot as plt
from datetime import datetime

import keras
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, TimeDistributed
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

from src.const_spec import *
from src import evalgen


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
UNIT_TEST = False
W_TRAIN_MODEL_WRONG_CLASS_NAMES = '[W] Check correctness of the class_names parameter that is passed to the' \
                                  ' train_model function.'


def create_model(model_type):
    """
    Function to create Neural Network model for recognizing TV News scenes.

    :param model_type: See const_spec class Architecture.
    :return: created model, optimizer, epochs and batch size
    """

    # Model architecture
    if model_type in Architecture.VGG16.FS.all:
        model = VGG16(include_top=False, input_shape=INPUT_SHAPE)

        x = model.layers[-1].output
        x = keras.layers.Flatten()(x)

        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)

    elif model_type in Architecture.VGG16.FDS.all:
        model = VGG16(include_top=False, input_shape=INPUT_SHAPE)

        x = model.layers[-1].output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)

        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
    else:
        raise Exception('[E] Required architecture does not exist!')

    # Optimizer
    if 'SGD' in model_type:
        optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    elif 'RMSprop' in model_type:
        optimizer = keras.optimizers.RMSprop(lr=0.0001, rho=0.8)
    else:
        raise Exception('[E] Optimizer not found!')

    # Number of epochs
    epochs = 20

    return model, optimizer, BatchSize[model_type], epochs


def train_model(model, train_data, test_data, epochs, batch_size, optimizer, output_path, class_names=None,
                tex_label=None, verbose=True):
    """
    Train model with the option to generate progress.

    :param model: Instance to the Neural Network to be trained.
    :param train_data: Tuple in format (data, labels). The data shape must be in input_shape format extended by
                       batch_shape, i.e. number of training data, the labels shape must be extended too.
    :param test_data: Same as train_data.
    :param epochs: The number of epochs for the model to be trained.
    :param batch_size: The number of samples that will be propagated through the network.
    :param optimizer: Keras optimizer instance.
    :param output_path: Path where is generated training history and where is saved trained model.
    :param class_names: Optional, the classification class names / labels.
    :param tex_label: Optional, the label of Confusion matrix that is created for the LaTeX document.
    :param verbose: Optional, display extended information.

    :return: History of training and output path.
    """
    # set the output directory name
    if output_path is not None:
        if not os.path.exists(output_path):
            raise Exception('[E] Output path is invalid!\n'
                            '- You are currently: ' + os.path.abspath('.') + '\n'
                            '- Your request was: ' + output_path + '\n')

        output_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # making a new directory by the date time
        output = output_path + ('\\' if output_path[-1] != '\\' else '') + output_dir + '\\'
        os.mkdir(output)

        with open(output + 'optimizer_config.txt', 'w') as f:
            f.write(f'Epochs: %d\n' % epochs)
            f.write(f'Optimizer: %s\n' % str(optimizer.__class__))
            f.write(str(optimizer.get_config()))

        with open(output + 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=120)
    else:
        output = ''  # [W]: Local variable 'output' might be referenced before assignment
    # if output_path is not None // set the output directory name

    # ==================================================================================================================
    # Part of training the model
    # ==================================================================================================================
    # setting the compiler for training and fitting the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(train_data[0], train_data[1], validation_data=test_data,
                        batch_size=batch_size, epochs=epochs, shuffle='batch', verbose=verbose)

    # ==================================================================================================================
    # Part of the output generation
    # ==================================================================================================================
    # generate an output if required
    if output_path is not None:
        # define the class names if not passed
        if class_names is None:
            class_names = CLASS_NAMES

        # ==============================================================================================================
        # Saving the model
        # ==============================================================================================================
        model.save(output + 'model.h5')

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
        # Creating the validation HTML file and LaTeX table
        # ==============================================================================================================
        if len(class_names) != model.get_output_shape_at(-1)[1]:
            print(W_TRAIN_MODEL_WRONG_CLASS_NAMES)
            print(f' - functions that are not called: %s' %
                  '(evalgen.create_tex_validation, evalgen.create_html_validation)')
            print(f' - class_names: %s' % class_names)
            print(f' - len(class_names) = %d' % len(class_names))
            print(f' - expected number of classes = %d' % model.get_output_shape_at(-1)[1])
        else:
            # computing the confusion matrix
            matrix, wrong_pred_vector = evalgen.get_confusion_matrix(model, test_data)

            # make dir for html file
            os.mkdir(output + 'HTML')

            # check if grad_cam can be created
            if len(model.get_input_shape_at(0)) == 5:
                grad_cam = False
            else:
                grad_cam = True
            # endif len(model.get_input_shape_at(0)) == 5 // check if grad_cam can be created

            # creating the LaTeX table and validation HTML file
            evalgen.create_tex_validation(matrix, class_names, output, label=tex_label, verbose=verbose)
            evalgen.create_html_validation(model, class_names, test_data, output + 'HTML', grad_cam=grad_cam,
                                           matrix_and_wrong_pred=(matrix, wrong_pred_vector),
                                           acc_loss=[output + 'acc.svg', output + 'loss.svg'], verbose=verbose)
        # endif len(class_names) != model.get_output_shape_at(-1)[1] // Creating the val HTML file and LaTeX table

        if not UNIT_TEST:
            print('\n\n')
            print(f'The output was successfully saved: %s' % output)
    # endif output_path is not None // generate the output if required

    return history
