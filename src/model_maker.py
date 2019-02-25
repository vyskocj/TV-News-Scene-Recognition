import h5py, os, glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
KERNEL_SIZE = (3, 3)
POOLING_SIZE = (2, 2)
BATCH_SIZE = 20
EPOCHS = 30
MODEL = "2build_n3_" + str(EPOCHS) + "e"
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
    model.add(Conv2D(32, kernel_size=KERNEL_SIZE,
                     activation='relu',
                     input_shape=shape, strides=3))
    model.add(Conv2D(32, kernel_size=KERNEL_SIZE,
                     activation='relu'))
    model.add(BatchNormalization())  # za kažou konv
    model.add(MaxPooling2D(pool_size=POOLING_SIZE,
                           strides=2))
    model.add(Dropout(0.20))

    model.add(Conv2D(64, kernel_size=KERNEL_SIZE,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=POOLING_SIZE,
                           strides=2))

    model.add(Conv2D(64, kernel_size=KERNEL_SIZE,
                     activation='relu'))
    model.add(BatchNormalization())  # za kažou konv
    model.add(MaxPooling2D(pool_size=POOLING_SIZE,
                           strides=2))
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(9, activation='softmax'))
    return model


def build(dataset_path):
    # první čílo _ zástupce (vybrat zátupce náhodně)
    # os.thisdir -> pro každý dir ho splitnu
    # sh.to copy
    # zjistit vyváženot tříd -> vyvážit jen ty, kde jich je hodně
    # load the dataset by path and get shape of images
    with h5py.File(dataset_path, 'r') as dataset:
        x_train, y_train, x_test, y_test, shape = read_dataset(dataset)
        model = create_model(shape)
        # model = keras.models.load_model('build_n1.h5')
        model.summary()

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      # sgd -> velmi náchylné na nastavení param : zkusím .01 -> pokud nic, tak .001 ..., momentum .9, zkusit nesterov=True
                      metrics=['accuracy'])  # rmsprop -> tu taky learning rate
        print(len(y_train[0]), len(y_test[0]))
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

        with open("..\\data\\model_out\\" + "out_" + MODEL + ".txt", "w") as f:
            f.write(str(history.history['acc']))
            f.write("\n")
            f.write(str(history.history['val_acc']))
            f.write("\n")
            f.write(str(history.history['loss']))
            f.write("\n")
            f.write(str(history.history['val_loss']))

build("..\\data\\dataset_balanced.h5")