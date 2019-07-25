import unittest
import os

from src import recognizer


# Set Unit Test state
recognizer.UNIT_TEST = True


class TestCreateModel(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
