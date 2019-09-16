from src.recognizer import *
from src.evalgen import *
from src.input_converter import *


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
        model = keras.models.load_model('..\\models\\LSTM\\model.h5')
        input_path = '..\\data\\dataset_test_lstm'
        output_path = '..\\output\\html_LSTM'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        decisions = create_html(model, CLASS_NAMES, input_path, output_path, files_together=True)

    elif phase == 3:
        model = keras.models.load_model('..\\models\\VGG16\\model.h5')
        input_path = '..\\data\\dataset_test'
        output_path = '..\\output\\html_VGG16'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        decisions = create_html(model, CLASS_NAMES_SH, input_path, output_path, files_together=True)

    elif phase == 4:
        model = keras.models.load_model('..\\models\\Own\\model.h5')
        input_path = '..\\data\\dataset_test'
        output_path = '..\\output\\html_Own'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        decisions = create_html(model, CLASS_NAMES_SH, input_path, output_path, files_together=True)

    elif phase == 5:
        dirs = os.listdir("..\\data\\dataset\\train")
        train_data_paths = ["..\\data\\dataset\\train\\" + name for name in dirs]
        test_data_paths = ["..\\data\\dataset\\test\\" + name for name in dirs]

        create_dataset(train_data_paths, test_data_paths, '..\\data\\dataset_ti.h5', img_shape=(180, 320, 3))

    elif phase == 6:
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.20))

        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.20))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(Dropout(0.40))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.summary()

        with h5py.File("..\\data\\dataset_ti.h5", 'r') as dataset:
            x_train = dataset[X_TRAIN]
            y_train = keras.utils.to_categorical(dataset[Y_TRAIN], NUM_CLASSES)

            x_test = dataset[X_TEST]
            y_test = keras.utils.to_categorical(dataset[Y_TEST], NUM_CLASSES)

            num_epochs = 40
            batch_size = 30

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')

