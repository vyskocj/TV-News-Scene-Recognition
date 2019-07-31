import h5py, os, glob
import keras
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
KERNEL_SIZE = (3, 3)
POOLING_SIZE = (2, 2)
BATCH_SIZE = 70
EPOCHS = 50
TYPE = "lowparams"
MODEL = "build_" + TYPE
DIR = "model_out"


def read_dataset(dataset):
    train_imgs = dataset["train_imgs"]
    test_imgs = dataset["test_imgs"]
    num_clusters = 0
    if "cluster_names" in dataset.keys():
        num_clusters = len(dataset["cluster_names"])
    else:
        num_clusters = max(dataset["test_imgs"][...]) + 1

    train_clusters = keras.utils.to_categorical(dataset["train_clusters"], num_clusters)
    test_clusters = keras.utils.to_categorical(dataset["test_clusters"], num_clusters)
    shape = get_shape(dataset)

    return train_imgs, train_clusters, test_imgs, test_clusters, shape


def get_shape(dataset):
    shape = (len(dataset["test_imgs"][0]),
             len(dataset["test_imgs"][0][0]),
             len(dataset["test_imgs"][0][0][0]))
    return shape


def create_model(shape):
    model = Sequential()
    model.add(Conv2D(24, kernel_size=KERNEL_SIZE, strides=1, activation='relu', input_shape=shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.20))

    model.add(Conv2D(60, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=POOLING_SIZE))

    model.add(Conv2D(150, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=POOLING_SIZE))
    model.add(Dropout(0.20))

    model.add(Conv2D(375, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=POOLING_SIZE))

    model.add(Conv2D(500, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=POOLING_SIZE))
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(750, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(9, activation='softmax'))

    return model


def build(dataset_path):
    # load the dataset by path and get shape of images
    with h5py.File(dataset_path, 'r') as dataset:
        x_train, y_train, x_test, y_test, shape = read_dataset(dataset)
        #model = keras.models.load_model('..\\data\\model_out\\build_21_vgg16.h5')
        #model.summary()

        #
        # načtení natrénovaného modelu a odstranění poslední vrstvy
        #
        model2 = Sequential()
        model2.add(TimeDistributed(VGG16(weights='imagenet', include_top=False), input_shape=(25,) + shape, trainable=False))
        model2.add(TimeDistributed(Flatten()))
        model2.add(TimeDistributed(Dense(1400, activation='relu'), trainable=False))
        model2.add(TimeDistributed(Dropout(0.50)))
        model2.add(TimeDistributed(Dense(9, activation='softmax')))


        print(model2.layers[0].get_config())

        model2.load_weights('..\\data\\my_model_weights.h5')
        model2.pop()

        model2.summary()

        #model2.save("..\\models\\part1.h5")

        #
        # model LSTM pro trénování
        #
        model2.add(LSTM(32, return_sequences=False, input_shape=(25, 1400)))
        model2.add(Dense(9, activation='softmax'))
        model2.summary()

        model = model2

        if False:
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.SGD(nesterov=True),
                          # sgd -> velmi náchylné na nastavení param : zkusím .01 -> pokud nic, tak .001 ..., momentum .9, zkusit nesterov=True
                          metrics=['accuracy'])  # rmsprop -> tu taky learning rate
        else:
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True),
                          # sgd -> velmi náchylné na nastavení param : zkusím .01 -> pokud nic, tak .001 ..., momentum .9, zkusit nesterov=True
                          metrics=['accuracy'])  # rmsprop -> tu taky learning rate

        history = model.fit(x_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            shuffle="batch")

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save("..\\data\\model_out\\" + MODEL + '.h5')
        print("acc")
        print(history.history["acc"])
        print("val_acc")
        print(history.history["val_acc"])

        import matplotlib.pyplot as plt
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("..\\data\\model_out\\" + "acc_" + MODEL + ".png")
        plt.clf()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("..\\data\\model_out\\" + "loss_" + MODEL + ".png")
        plt.clf()

        with open("..\\data\\model_out\\" + "out_" + MODEL + ".txt", "w") as f:
            f.write(str(history.history['acc']))
            f.write("\n")
            f.write(str(history.history['val_acc']))
            f.write("\n")
            f.write(str(history.history['loss']))
            f.write("\n")
            f.write(str(history.history['val_loss']))


build("..\\data\\dataset.h5")