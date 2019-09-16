import h5py
import random
import cv2
import os
import glob

from src.const_spec import *


def create_dataset(train_data_paths, test_data_paths, output_filename, img_shape=None, shuffle=True, balance=None,
                   divide_by_255=True, verbose=True):
    """
    Create a new dataset from directory files.

    :param train_data_paths: List in format: [path_to_first_class_data, path_to_second_class_data, ...], where the paths
                             are to the folders that the set of all training data are stored.
    :param test_data_paths: Same as train_data_paths but the paths are to the testing data.
    :param output_filename: The name of output file.
    :param img_shape: Optional, set the image size in the dataset.
    :param shuffle: Optional, the training data should be shuffled before fitting the model.
    :param balance: Optional, for correct usage the training data should be in form: scene_frame.png.
    :param divide_by_255: Optional, images are divided by 255 if True is given.
    :param verbose: Optional runtime printing.
    :return: OK = 0
    """

    if verbose:
        print("The dataset is being created: ")

    # Check if balance is defined
    if balance is None:
        balance = [False] * len(train_data_paths)

    # Get path of all images that have to be included to the dataset
    train_imgs, train_labels, num_train_imgs = get_img_paths(train_data_paths, balance)
    test_imgs, test_labels, num_test_imgs = get_img_paths(test_data_paths)

    # Check image shape: if img_shape is None, keep the original shape
    if img_shape is None:
        img = cv2.imread(train_imgs[0])
        img_shape = img.shape

    # The input data to the Neural Network should be randomly distributed
    if shuffle is True:
        train_imgs, train_labels = shuffle_lists(train_imgs, train_labels)

    # Creating dataset with passed data
    with h5py.File(output_filename, 'w') as output_file:
        # Defining dataset for the input data
        output_file.create_dataset(X_TRAIN, shape=(num_train_imgs,) + img_shape, dtype="float32")
        output_file.create_dataset(X_TEST, shape=(num_test_imgs,) + img_shape, dtype="float32")

        # Adding Neural Network output labels
        output_file.create_dataset(Y_TRAIN, shape=(num_train_imgs,), data=train_labels, dtype="uint8")
        output_file.create_dataset(Y_TEST, shape=(num_test_imgs,), data=test_labels, dtype="uint8")

        # Adding data to output_file
        if verbose:
            print("- Adding training data:")
        add_imgs(train_imgs, output_file[X_TRAIN], img_shape, divide_by_255, verbose)

        if verbose:
            print("- Adding testing data:")
        add_imgs(test_imgs, output_file[X_TEST], img_shape, divide_by_255, verbose)
    # endwith h5py.File(output_filename, 'w') as output_file // Creating dataset with passed data

    # Print that dataset was successfully created
    if verbose:
        print("Dataset was successfully created.")

    return OK


def shuffle_dataset(input_dataset, output_dataset, input_keys, output_keys=None, divide_by_255=True, verbose=True):
    """
    Create new dataset with shuffled data which can be used for fitting the Neural Network model.

    :param input_dataset: Path to the one given dataset or paths (list / tuple) to the several datasets.
    :param output_dataset: Path and name of the output dataset.
    :param input_keys: The pairs (data, labels) in the input dataset(s). If more than one dataset is given and each
                       dataset has different key names, the correct format is: (pairs_1, pairs_2, ...)
                       where e.g. pairs_1 = ("train_data", "train_labels"), pairs_2 = ("input", "output"), ...
    :param output_keys: Optional when only one dataset is given. It defines pairs for the new dataset.
    :param divide_by_255: Optional, frames are divided by 255 if True is given.
    :param verbose: Optional runtime printing.
    :return: If success: OK = 0, if fail: INVALID_INPUT_PARAMETER = -2
    """

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
                    if len(data_shape) == 5:
                        # dataset for LSTM
                        num_frames = data_shape[1]

                        # save data to the dataset file
                        for k, s in enumerate(shuffle_list):
                            # print progress every 50th set if print is required
                            if k % 50 == 0 and verbose:
                                print(" [" + str(k) + "/" + str(num_data) + "] set")

                            for l in range(0, num_frames):
                                new_dataset[output_keys[d][j_data]][k, l, ...] = \
                                    old_dataset[input_keys[d][i][j_data]][s, l, ...] / (255.0 if divide_by_255 else 1.0)
                            # endfor l in range(0, num_frames)

                            labels.append(old_dataset[input_keys[d][i][j_lbls]][s, ...])
                        # endfor k, s in enumerate(shuffle_list) // save data to the dataset file
                    else:
                        # dataset for time invariant distributed model
                        # save data, i.e. picture-by-picture to the new dataset file
                        for k, s in enumerate(shuffle_list):
                            # print progress every 50th set if print is required
                            if k % 50 == 0 and verbose:
                                print(" [" + str(k) + "/" + str(num_data) + "] set")

                            new_dataset[output_keys[d][j_data]][k, l, ...] = \
                                old_dataset[input_keys[d][i][j_data]][s, ...] / (255.0 if divide_by_255 else 1.0)

                            labels.append(old_dataset[input_keys[d][i][j_lbls]][s, ...])
                        # endfor k, s in enumerate(shuffle_list) // save data to the new dataset file
                    # endif len(data_shape) == 5 // deciding whether it is an LSTM dataset

                    # save labels to the new dataset
                    new_dataset.create_dataset(output_keys[d][j_lbls], shape=lbls_shape, data=labels, dtype="uint8")
                # endfor i in range(0, len(input_keys[d])) // for every pair (data, labels) in the old dataset
            # endwith h5py.File(dataset, 'r') as old_dataset // open required dataset
        # endfor d, dataset in enumerate(input_dataset) // for each required dataset
    # endwith h5py.File(output_dataset, 'w') as new_dataset // create a new dataset

    return OK


def get_img_paths(data, balance=None):
    data_paths = list()   # list for all images
    data_labels = list()  # list of all classifications

    # Assumption that directories are given - each directory is equal to one label
    for index, path in enumerate(data):
        # balance images if required
        if balance is not None and balance[index] is True:
            # get the img names
            img_names = [name.split("\\")[-1] for name in img_paths]

            # create dictionary for images and throw all paths
            dict_imgs = dict()
            img_paths = list()

            # drop out images from the same scenes - balancing
            for i, img in enumerate(img_names):
                # split the img name, i.e. scene1_frame5.jpg
                img_names[i] = img_names[i].split("_")

                # add img into dictionary (by type of scene)
                if img_names[i][0] in dict_imgs.keys():
                    dict_imgs[img_names[i][0]].append(img_names[i][1])
                else:
                    dict_imgs[img_names[i][0]] = [img_names[i][1], ]
            # endfor i, img in enumerate(img_names) // drop out images from the same scenes

            # Add the 1 random frame from each scene
            [img_paths.append(path + "\\" + img + "_" + random.choice(dict_imgs[img])) for img in dict_imgs.keys()]

            # set a new size
            size = len(img_paths)
        else:
            # get list of paths and count them
            img_paths = glob.glob(os.path.join(path, "*"))
            size = len(img_paths)
        # endif balance is not None and balance[index] is True // balance images if required

        # merge lists
        data_paths += img_paths
        data_labels += [index] * size
    # endfor index, path in enumerate(data) // Assumption that directories are given

    return data_paths, data_labels, len(data_paths)


def shuffle_lists(list1, list2):

    # Packing pairs - (input, output)
    combined = list(zip(list1, list2))

    # Randomize the list
    random.shuffle(combined)

    # Unpack pairs
    list1[:], list2[:] = zip(*combined)

    return list1, list2


def add_imgs(data, output, shape, divide_by_255=True, verbose=True):

    num_data = len(data)

    # read all image paths
    for index, path in enumerate(data):
        # The status is printed every 50 saved images
        if verbose and index % 50 == 0:
            print(" [" + str(index) + "/" + str(num_data) + "] image")

        # Load image and reshape it
        img = cv2.imread(path)
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

        # Add image into output_file
        output[index, ...] = img

        # Transform image between [0, 1]
        if divide_by_255:
            output[index, ...] /= 255.0
    # endfor index, path in enumerate(data) // read all image paths


# TODO: přehodit do mainu
# shuffle_dataset(('..\\data\\dataset_lstm\\scene_recog_test.h5', '..\\data\\dataset_lstm\\scene_recog_train.h5'), '..\\data\\dataset_lstm\\scene_recog.h5', ('labels', 'data'), [('test_labels', 'test_data'), ('train_labels', 'train_data')])
