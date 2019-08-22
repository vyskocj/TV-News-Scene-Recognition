from src.recognizer import *
from src.evalgen import *


# TODO: předělat s cílem lepší interakce
if __name__ == '__main__':
    phase = int(input('Which phase do you want to train?\n'))

    if phase == 0:
        model = create_model(INPUT_SHAPE, NUM_CLASSES, use_lstm=False)

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 40
            batch_size = 50

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')

    elif phase == 1:
        model = create_model(INPUT_SHAPE_TD, NUM_CLASSES, use_lstm=True, first_part_trainable=False,
                             first_part_weights='..\\data\\my_model_weights.h5')

        with h5py.File("..\\data\\dataset_lstm\\scene_recog.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 40
            batch_size = 5

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')

    elif phase == 2:
        model = keras.models.load_model('..\\output\\2019-07-28_21-15-23_LSTM_1st_TRY\\model.h5')
        input_path = '..\\data\\dataset_test_lstm'
        output_path = '..\\output\\html_test'
        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        create_html(model, CLASS_NAMES_SH, input_path, output_path)
