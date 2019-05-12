import h5py, os, glob
import keras
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
KERNEL_SIZE = (3, 3)
POOLING_SIZE = (2, 2)
BATCH_SIZE = 70
EPOCHS = 200
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
    # první čílo _ zástupce (vybrat zátupce náhodně)
    # os.thisdir -> pro každý dir ho splitnu
    # sh.to copy
    # zjistit vyváženot tříd -> vyvážit jen ty, kde jich je hodně
    # load the dataset by path and get shape of images
    for i in range(9, 10):
        with h5py.File(dataset_path, 'r') as dataset:
            x_train, y_train, x_test, y_test, shape = read_dataset(dataset)
            # model = create_model(shape)
            # model = keras.models.load_model('..\\data\\model_out\\build_vgg16.h5')
            # model.summary()

            if i in [5, 6, 7, 8]:
                model_app = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
                #BATCH_SIZE = 10
                TYPE = "resnet50"
            else:
                model_app = VGG16(weights='imagenet', include_top=False, input_shape=shape)
                BATCH_SIZE = 48
                #TYPE = "vgg16"
            #MODEL = "build_" + str(i) + "_" + TYPE

            #for layer in model_app.layers[:-5]:
            #    layer.trainable = False

            model = Sequential()
            model.add(model_app)
            model.add(Flatten())

            if i == 1 or i == 5:
                model.add(Dense(800, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(9, activation='softmax'))
            elif i == 2 or i == 6:
                model.add(Dense(1400, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(9, activation='softmax'))
            elif i == 3 or i == 7:
                model.add(Dense(800, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(1400, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(9, activation='softmax'))
            elif i == 4 or i == 8:
                model.add(Dense(1400, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(800, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(9, activation='softmax'))
            elif i == 9:
                model.add(Dense(2000, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(2000, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(1500, activation='relu'))
                model.add(Dense(9, activation='softmax'))
            elif i == 21:
                model = keras.models.load_model('..\\data\\model_out\\build_2_vgg16.h5')
            elif i == 22:
                model.add(Dense(1500, activation='relu'))
                model.add(Dropout(0.50))
                model.add(Dense(1500, activation='relu'))
                model.add(Dense(9, activation='softmax'))

            model = create_model(shape)
            model.summary()


            if len(y_train) > 10000:
                EPOCHS = 200
            elif len(y_train) > 5000:
                EPOCHS = 80
            elif len(y_train) > 1000:
                EPOCHS = 300
            else:
                EPOCHS = 800

            if i in [21]:
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