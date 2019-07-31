from src.const_spec import *
from src.recognizer import *

if __name__ == '__main__':
    phase = 10
    if phase == 0:
        pass
    elif phase == 10:
        model = create_model(INPUT_SHAPE_TD, NUM_CLASSES, use_lstm=True, first_part_trainable=False,
                             first_part_weights='..\\data\\my_model_weights.h5')

        with h5py.File("..\\data\\dataset_lstm\\scene_receg.h5", 'r') as dataset:
            x_train = dataset["train_data"]
            y_train = keras.utils.to_categorical(dataset["train_labels"], NUM_CLASSES)

            x_test = dataset["test_data"]
            y_test = keras.utils.to_categorical(dataset["test_labels"], NUM_CLASSES)

            num_epochs = 50
            batch_size = 5

            train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                        keras.optimizers.SGD(lr=.001, momentum=.9, nesterov=True), output_path='..\\output')
