import unittest
import os
import keras

import numpy as np

from src import recognizer


# Set Unit Test state
recognizer.UNIT_TEST = True


class TestRecognizer(unittest.TestCase):
    def test_create_model_1(self):
        # Test 1 - First part of model: VGG16 with fully connected layers
        input_shape = (32, 34, 3)
        num_classes = 5
        model = recognizer.create_model(input_shape, num_classes, False, None)

        if 'test_data' not in os.listdir():
            os.mkdir('test_data')
        if 'test_weights.h5' not in os.listdir('test_data'):
            model.save_weights('test_data\\test_weights.h5')

        # ==============================================================================================================
        # TEST CONDITIONS

        # First layer has to be VGG16
        self.assertEqual(model.layers[0].get_config()['name'],
                         'vgg16')
        # Shape of first layer has to be set correctly
        self.assertEqual(model.layers[0].get_config()['layers'][0]['config']['batch_input_shape'],
                         (None, ) + input_shape)

        # Last layer has to be dense
        self.assertEqual(model.layers[-1].get_config()['name'].split('_')[0],
                         'dense')
        # Classification has to be set correctly
        self.assertEqual(model.layers[-1].get_config()['units'],
                         num_classes)

    def test_create_model_2(self):
        # Test 2 - First part of model is successfully trained: Load the treined weights
        input_shape = (32, 34, 3)
        num_classes = 5
        model = recognizer.create_model(input_shape, num_classes, False, 'test_data\\test_weights.h5')

        # ==============================================================================================================
        # TEST CONDITIONS

        # First layer has to be VGG16
        self.assertEqual(model.layers[0].get_config()['name'],
                         'vgg16')
        # Shape of first layer has to be set correctly
        self.assertEqual(model.layers[0].get_config()['layers'][0]['config']['batch_input_shape'],
                         (None, ) + input_shape)

        # Last layer has to be dense
        self.assertEqual(model.layers[-1].get_config()['name'].split('_')[0],
                         'dense')
        # Classification has to be set correctly
        self.assertEqual(model.layers[-1].get_config()['units'],
                         num_classes)

    def test_create_model_3(self):
        # Test 3 - Use the 2nd part of model too with trained weights
        input_shape = (2, 32, 34, 3)
        num_classes = 5
        model = recognizer.create_model(input_shape, num_classes, True, 'test_data\\test_weights.h5')

        # ==============================================================================================================
        # TEST CONDITIONS

        # First layer has to be Time-Distributed VGG16
        self.assertEqual(model.layers[0].get_config()['name'].split('_')[0:2],
                         ['time', 'distributed'])
        self.assertEqual(model.layers[0].get_config()['layer']['config']['name'],
                         'vgg16')
        # Shape of first layer has to be set correctly
        self.assertEqual(model.layers[0].get_config()['batch_input_shape'],
                         (None, ) + input_shape)

        # There shall be no classification layer at the end of the first part
        self.assertEqual(model.layers[-3].get_config()['layer']['config']['name'].split('_')[0],
                         'dropout')

        # In the second part has to be LSTM network
        self.assertEqual(model.layers[-2].get_config()['name'].split('_')[0],
                         'lstm')

        # Last layer has to be dense
        self.assertEqual(model.layers[-1].get_config()['name'].split('_')[0],
                         'dense')
        # Classification has to be set correctly
        self.assertEqual(model.layers[-1].get_config()['units'],
                         num_classes)

    def test_train_model(self):
        # Create any model
        input_shape = (32, 34, 3)
        num_classes = 5
        model = recognizer.create_model(input_shape, num_classes, False, None)

        # Random train and validation data
        x_train = np.asarray([np.random.rand(input_shape[0], input_shape[1], input_shape[2])] * 10)
        y_train = keras.utils.to_categorical(list(range(0, 5)) * 2, num_classes)

        x_test = np.asarray([np.random.rand(input_shape[0], input_shape[1], input_shape[2])] * 5)
        y_test = keras.utils.to_categorical(list(range(0, 5)), num_classes)

        num_epochs = 3
        batch_size = 10

        # Test function
        his, path = recognizer.train_model(model, (x_train, y_train), (x_test, y_test), num_epochs, batch_size,
                                           keras.optimizers.SGD(), output_path='test_data')

        # ==============================================================================================================
        # TEST CONDITIONS

        # The output has been successfully saved
        self.assertNotEqual(path, None)

        # The history of training has been saved
        self.assertEqual('history.xls' in os.listdir(path), True)

        # The final model has been saved
        self.assertEqual('model.h5' in os.listdir(path), True)


if __name__ == '__main__':
    unittest.main()
