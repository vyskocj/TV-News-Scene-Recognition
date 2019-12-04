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
        decisions = create_html(model, CLASS_NAMES, input_path, output_path, portable=True)








    elif phase == 3:
        # model = keras.models.load_model('..\\models\\VGG16\\model.h5')
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(180, 320, 3)))
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(1400, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(9, activation='softmax'))
        model.load_weights("..\\data\\weights_vgg16.h5")

        input_path = '..\\data\\dataset_test'
        output_path = '..\\output\\html_VGG16\\html_pred'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        create_html(model, CLASS_NAMES_SH, input_path, output_path, portable=True, grad_cam=True)

    elif phase == 331:

        model = keras.models.load_model('..\\models\\LSTM\\model.h5')

        input_path = '..\\data\\dataset_test_lstm_2'
        output_path = '..\\output\\html_LSTM\\html_pred'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        decisions = create_html(model, CLASS_NAMES, input_path, output_path, portable=True)

        output_path = '..\\output\\html_LSTM\\html_valid'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with h5py.File("..\\data\\dataset_lstm\\scene_recog.h5", 'r') as dataset:
            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            create_html_validation(model, CLASS_NAMES, (x_test, y_test), output_path, grad_cam=False)

        # model = keras.models.load_model('..\\models\\VGG16\\model.h5')
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(180, 320, 3)))
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(1400, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(9, activation='softmax'))
        model.load_weights("..\\data\\weights_vgg16.h5")

        output_path = '..\\output\\html_VGG16\\html_valid'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            create_html_validation(model, CLASS_NAMES, (x_test, y_test), output_path, grad_cam=True)









    elif phase == 4:
        model = keras.models.load_model('..\\models\\Own\\model.h5')
        input_path = '..\\data\\dataset_test'
        output_path = '..\\output\\html_Own'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        decisions = create_html(model, CLASS_NAMES, input_path, output_path, portable=True, grad_cam=True)

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

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset[X_TRAIN]
            y_train = keras.utils.to_categorical(dataset[Y_TRAIN], NUM_CLASSES)

            x_test = dataset[X_TEST]
            y_test = keras.utils.to_categorical(dataset[Y_TEST], NUM_CLASSES)

            num_epochs = 40
            batch_size = 125

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')

    elif phase == 10:
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(180, 320, 3)))
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(1400, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(9, activation='softmax'))
        model.load_weights("..\\data\\weights_vgg16.h5")

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            #matrix, _ = get_confusion_matrix+(model, (x_test, y_test))
            #print(matrix)

            #create_tex_validation(matrix, CLASS_NAMES, '..\\output')
            create_html_validation(model, CLASS_NAMES, (x_test, y_test), '..\\output\\test', grad_cam=True)

    elif phase == 11:
        model = keras.models.load_model('..\\models\\LSTM\\model.h5')

        with h5py.File("..\\data\\dataset_lstm\\scene_recog.h5", 'r') as dataset:
            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            create_html_validation(model, CLASS_NAMES, (x_test, y_test), '..\\output\\test', grad_cam=False)

    elif phase == 20:
        vgg = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=INPUT_SHAPE))
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add_weight(vgg.get_weights())

    elif phase == 21:
        from keras.models import Model

        model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

        x = model.layers[-1].output
        x = Flatten()(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
        model.summary()

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset[X_TRAIN]
            y_train = keras.utils.to_categorical(dataset[Y_TRAIN], NUM_CLASSES)

            x_test = dataset[X_TEST]
            y_test = keras.utils.to_categorical(dataset[Y_TEST], NUM_CLASSES)

            num_epochs = 40
            batch_size = 75

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.90, nesterov=True), output_path='..\\output')

    elif phase == 22:
        model = keras.models.load_model('..\\output\\VGG16_lowparam\\model.h5')
        input_path = '..\\data\\dataset_test'
        output_path = '..\\output\\VGG16_lowparam\\html_pred\\'

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # TODO: volat na lepším místě, generování do outputu po natrénování sítě
        decisions = create_html(model, CLASS_NAMES, input_path, output_path, portable=True, grad_cam=True)

    elif phase == 23:
        from keras.models import Model
        from keras.layers import LSTM, TimeDistributed

        model = keras.models.Sequential()
        model.add(
            TimeDistributed(
                VGG16(weights='imagenet', include_top=False),
                input_shape=INPUT_SHAPE_TD,
                trainable=False
            )
        )
        x = model.layers[-1].output
        x = TimeDistributed(Flatten())(x)
        predictions = TimeDistributed(Dense(NUM_CLASSES, activation='softmax'))(x)
        model = Model(inputs=model.input, outputs=predictions)

        model.set_weights(keras.models.load_model('..\\output\\VGG16_lowparam\\model.h5').get_weights())

        model.pop()

        model.add(LSTM(32, input_shape=(NUM_FRAMES, model.layers[-1].shape())))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

    elif phase == 30:
        dirs = os.listdir("..\\data\\dataset_lstm\\test")
        test_data_paths = ["..\\data\\dataset_lstm\\test\\" + name for name in dirs]

        create_dataset('..\\data\\dataset_lstm\\test_prima.h5', test_data_paths=test_data_paths, img_shape=(25, 180, 320, 3), lstm=True)

    elif phase == 31:
        model = keras.models.load_model('..\\models\\LSTM\\model.h5')

        with h5py.File('..\\data\\dataset_lstm\\test_prima.h5', 'r') as dataset:
            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            create_html_validation(model, CLASS_NAMES, (x_test, y_test), '..\\output\\test_lstm', grad_cam=False)

    elif phase == 32:
        dirs = os.listdir("..\\data\\dataset\\test")
        test_data_paths = ["..\\data\\dataset\\test\\" + name for name in dirs]

        create_dataset('..\\data\\dataset\\test_prima.h5', test_data_paths=test_data_paths, img_shape=(180, 320, 3), lstm=False)

    elif phase == 33:
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(180, 320, 3)))
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(1400, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(9, activation='softmax'))
        model.load_weights("..\\data\\weights_vgg16.h5")

        with h5py.File('..\\data\\dataset\\test_prima.h5', 'r') as dataset:
            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            create_html_validation(model, CLASS_NAMES, (x_test, y_test), '..\\output\\test_vgg16', grad_cam=True)

    elif phase == 50:
        model = keras.applications.InceptionResNetV2(include_top=False, input_shape=(180, 320, 3))

        for l in model.layers:
            l.trainable = False

        x = model.layers[-1].output
        x = keras.layers.GlobalAveragePooling2D()(x)

        predictions = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = keras.models.Model(inputs=model.input, outputs=predictions)

        model.summary()

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 5
            batch_size = 30

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')

    elif phase == 51:
        model = keras.models.load_model('..\\output\\2019-11-09_23-46-00\\model.h5')

        for l in model.layers:
            l.trainable = True

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 20
            batch_size = 30

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.00001, momentum=.9), output_path='..\\output')

    elif phase == 52:
        model = keras.models.load_model('..\\output\\2019-11-11_07-03-15\\model.h5')

        for l in model.layers:
            l.trainable = True

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 20
            batch_size = 20

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.Adam(0.0001), output_path='..\\output')

    elif phase == 53:
        model = keras.applications.InceptionResNetV2(include_top=False, input_shape=(180, 320, 3))

        x = model.layers[-1].output
        x = keras.layers.Flatten()(x)

        predictions = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = keras.models.Model(inputs=model.input, outputs=predictions)

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 20
            batch_size = 20

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')

    elif phase == 54:
        model = keras.applications.VGG16(include_top=False, input_shape=(180, 320, 3))

        x = model.layers[-1].output
        x = keras.layers.Flatten()(x)

        predictions = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = keras.models.Model(inputs=model.input, outputs=predictions)

        with h5py.File("..\\data\\dataset\\scene_recog.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 20
            batch_size = 20

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')

    elif phase == 55:
        with h5py.File('..\\data\\dataset\\test_prima.h5', 'r') as dataset:
            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            model = keras.models.load_model('..\\output\\2019-11-14_06-05-36_GAP\\model.h5')
            create_html_validation(model, CLASS_NAMES, (x_test, y_test), '..\\output\\TEST_RUN', grad_cam=False,
                                   acc_loss=['..\\output\\2019-11-14_06-05-36_GAP\\acc.svg', '..\\output\\2019-11-14_06-05-36_GAP\\loss.svg'])

            model = keras.models.load_model('..\\output\\2019-11-14_23-48-35_VGG16\\model.h5')
            create_html_validation(model, CLASS_NAMES, (x_test, y_test), '..\\output\\TEST_RUN2', grad_cam=False)

    elif phase == 60:

        model = VGG16(include_top=False, input_shape=INPUT_SHAPE)

        x = model.layers[-1].output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)

        predictions = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)

        model.summary()
