import h5py
import random

from src.const_spec import *


def shuffle_dataset(input_dataset, output_dataset, input_keys, output_keys=None, divide_by_255=True, verbose=True):

    # check the input_dataset parameter
    if type(input_dataset) is str:
        # only one dataset is given by the parameter
        input_dataset = [input_dataset]
    # endif type(input_dataset) is str // check the input_dataset parameter

    # check the input_keys parameter
    if type(input_keys[0]) is not tuple or type(input_keys[0]) is not list:
        # only one pair of input keys is passed to the function
        input_keys = [[input_keys]] * len(input_dataset)

        # check if the output keys can be created correctly
        if output_keys is None and len(input_dataset) != 1:
            print('[E] Define parameter output_keys for function shuffle_dataset!')
            return INVALID_INPUT_PARAMETER
        # endif output_keys is None and len(input_dataset) != 1 // check if the output keys can be created correctly
    # endif type(input_keys[0]) is not tuple or type(input_keys[0]) is not list // check the input_keys parameter

    # automatic creations of output keys
    if output_keys is None:
        output_keys = input_keys

    # create a new dataset
    with h5py.File(output_dataset, 'w') as new_dataset:
        # for each required dataset
        for d, dataset in enumerate(input_dataset):
            if verbose:
                print(f'Dataset: %s' % dataset)

            # open required dataset
            with h5py.File(dataset, 'r') as old_dataset:
                # for every pair (data, labels) in the old dataset
                for i in range(0, len(input_keys[d])):
                    # initialize indexes for data and labels
                    j_data = 0
                    j_lbls = 1
                    # assumption that the label has a shape length equal to two
                    if len(old_dataset[input_keys[d][i][j_data]].shape) == 2:
                        # the indexes are swapped
                        j_data = 1
                        j_lbls = 0

                    # initialize data and labels shape variables
                    data_shape = old_dataset[input_keys[d][i][j_data]].shape
                    lbls_shape = old_dataset[input_keys[d][i][j_lbls]].shape

                    # create dataset for data
                    new_dataset.create_dataset(output_keys[d][j_data], shape=data_shape, dtype="float32")

                    # create a random index list
                    num_data = data_shape[0]
                    shuffle_list = list(range(0, num_data))
                    random.shuffle(shuffle_list)

                    # deciding whether it is an LSTM dataset
                    labels = list()
                    # TODO: dokomentovat, použít parametr divide_by_255
                    if len(data_shape) == 5:  # LSTM dataset
                        num_frames = data_shape[1]

                        # save data to the dataset file
                        for k, s in enumerate(shuffle_list):
                            # print progress every 50th set if required
                            if k % 50 == 0 and verbose:
                                print(" [" + str(k) + "/" + str(num_data) + "] set")

                            for l in range(0, num_frames):
                                new_dataset[output_keys[d][j_data]][k, l, ...] = old_dataset[input_keys[d][i][j_data]][s, l, ...] / 255.0

                            labels.append(old_dataset[input_keys[d][i][j_lbls]][s, ...])

                    else:                     # non time distributed
                        for k, s in enumerate(shuffle_list):
                            print(" [" + str(k) + "/" + str(num_data) + "] set")
                            new_dataset[output_keys[d][j_data]][k, l, ...] = old_dataset[input_keys[d][i][j_data]][s, ...] / 255.0

                            labels.append(old_dataset[input_keys[d][i][j_lbls]][s, ...])

                    new_dataset.create_dataset(output_keys[d][j_lbls], shape=lbls_shape, data=labels, dtype="uint8")

# TODO: přehodit do mainu
# shuffle_dataset(('..\\data\\dataset_lstm\\scene_recog_test.h5', '..\\data\\dataset_lstm\\scene_recog_train.h5'), '..\\data\\dataset_lstm\\scene_recog.h5', ('labels', 'data'), [('test_labels', 'test_data'), ('train_labels', 'train_data')])
