import h5py
import random
import cv2
import os
import glob
import time
import numpy as np

from src.const_spec import *


def create_dataset(output_filename, train=None, valid=None, test=None, img_shape=None,
                   lstm=False, shuffle=True, balance=None, normalize=True, verbose=True):
    """
    Create a new dataset from directory files.

    :param output_filename: Name of the output file.
    :param train: Path to the training data.
    :param valid: Path to the validation data.
    :param test: Path to the testing data.
    :param img_shape: Optional, set the image size in the dataset.
    :param lstm: Optional, the dataset is time-independent.
    :param shuffle: Optional, the training data should be shuffled before fitting the model.
    :param balance: Optional, for correct usage the training data should be in form: scene_frame.png.
    :param normalize: Optional, images are divided by 255 if True is given.
    :param verbose: Optional runtime printing.
    """

    if verbose:
        print("[I] The dataset is being created")

    # Check if balance is defined
    if balance is None and train is not None:
        balance = [False] * len(train)

    # Get path of all images that have to be included to the dataset
    if train is not None:
        train_imgs = [os.path.join(train, name) for name in os.listdir(train)]
        train_imgs, train_labels, num_train_imgs = get_img_paths(train_imgs, balance, lstm=lstm)
        num_labels = max(train_labels)
    else:
        train_imgs, train_labels, num_train_imgs, num_labels = None, None, None, 0  # Dummy variables

    if valid is not None:
        valid_imgs = [os.path.join(valid, name) for name in os.listdir(valid)]
        valid_imgs, valid_labels, num_valid_imgs = get_img_paths(valid_imgs, lstm=lstm)
        if num_labels == 0:
            num_labels = max(valid_labels)
    else:
        valid_imgs, valid_labels, num_valid_imgs = None, None, None   # Dummy variables

    if test is not None:
        test_imgs = [os.path.join(test, name) for name in os.listdir(test)]
        test_imgs, test_labels, num_test_imgs = get_img_paths(test_imgs, lstm=lstm)
        if num_labels == 0:
            num_labels = max(test_labels)
    else:
        test_imgs, test_labels, num_test_imgs = None, None, None   # Dummy variables

    # Check image shape: if img_shape is None, keep the original shape
    if img_shape is None:
        if train is not None:
            img = cv2.imread(train_imgs[0][0] if lstm else train_imgs[0])
            img_shape = ((len(train_imgs[0]),) if lstm else ()) + img.shape
        elif valid is not None:
            img = cv2.imread(valid_imgs[0][0] if lstm else valid_imgs[0])
            img_shape = ((len(valid_imgs[0]),) if lstm else ()) + img.shape
        elif test is not None:
            img = cv2.imread(test_imgs[0][0] if lstm else test_imgs[0])
            img_shape = ((len(test_imgs[0]),) if lstm else ()) + img.shape
        else:
            raise Exception('[E] No data was passed to the function.')

    # The input data to the Neural Network should be randomly distributed
    if shuffle is True and train is not None:
        train_imgs, train_labels = shuffle_lists(train_imgs, train_labels)

    # Creating dataset with passed data
    with h5py.File(output_filename, 'w') as output_file:
        dtype_img = 'float16' if normalize else 'uint8'
        dtype_lbl = 'uint8' if num_labels < 255 else 'uint16'

        # Defining dataset for the input data
        if train is not None:
            output_file.create_dataset(X_TRAIN, shape=(num_train_imgs,) + img_shape, dtype=dtype_img)
        if valid is not None:
            output_file.create_dataset(X_VALID, shape=(num_valid_imgs,) + img_shape, dtype=dtype_img)
        if test is not None:
            output_file.create_dataset(X_TEST, shape=(num_test_imgs,) + img_shape, dtype=dtype_img)

        # Adding Neural Network output labels
        if train is not None:
            output_file.create_dataset(Y_TRAIN, shape=(num_train_imgs,), data=train_labels, dtype=dtype_lbl)
        if valid is not None:
            output_file.create_dataset(Y_VALID, shape=(num_valid_imgs,), data=valid_labels, dtype=dtype_lbl)
        if test is not None:
            output_file.create_dataset(Y_TEST, shape=(num_test_imgs,), data=test_labels, dtype=dtype_lbl)

        # Adding data to output_file
        if train is not None:
            if verbose:
                print("- Training data:")
            add_imgs(train_imgs, output_file[X_TRAIN], img_shape, normalize, verbose, lstm=lstm)

        if valid is not None:
            if verbose:
                print("- Validation data:")
            add_imgs(valid_imgs, output_file[X_VALID], img_shape, normalize, verbose, lstm=lstm)

        if test is not None:
            if verbose:
                print("- Testing data:")
            add_imgs(test_imgs, output_file[X_TEST], img_shape, normalize, verbose, lstm=lstm)
    # endwith h5py.File(output_filename, 'w') as output_file // Creating dataset with passed data

    # Print that dataset was successfully created
    if verbose:
        print("[I] Dataset was successfully created.")


def shuffle_dataset(input_dataset, output_dataset, input_keys, output_keys=None, normalize=True, verbose=True):
    """
    Create new dataset with shuffled data which can be used for fitting the Neural Network model.

    :param input_dataset: Path to the one given dataset or paths (list / tuple) to the several datasets.
    :param output_dataset: Path and name of the output dataset.
    :param input_keys: The pairs (data, labels) in the input dataset(s). If more than one dataset is given and each
                       dataset has different key names, the correct format is: (pairs_1, pairs_2, ...)
                       where e.g. pairs_1 = ("train_data", "train_labels"), pairs_2 = ("input", "output"), ...
    :param output_keys: Optional when only one dataset is given. It defines pairs for the new dataset.
    :param normalize: Optional, frames are divided by 255 if True is given.
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
                                    old_dataset[input_keys[d][i][j_data]][s, l, ...] / (255.0 if normalize else 1.0)
                            # endfor l in range(0, num_frames)

                            labels.append(old_dataset[input_keys[d][i][j_lbls]][s, ...])
                        # endfor k, s in enumerate(shuffle_list) // save data to the dataset file
                    else:
                        # dataset for time independent distributed model
                        # save data, i.e. picture-by-picture to the new dataset file
                        for k, s in enumerate(shuffle_list):
                            # print progress every 50th set if print is required
                            if k % 50 == 0 and verbose:
                                print(" [" + str(k) + "/" + str(num_data) + "] set")

                            new_dataset[output_keys[d][j_data]][k, ...] = \
                                old_dataset[input_keys[d][i][j_data]][s, ...] / (255.0 if normalize else 1.0)

                            labels.append(old_dataset[input_keys[d][i][j_lbls]][s, ...])
                        # endfor k, s in enumerate(shuffle_list) // save data to the new dataset file
                    # endif len(data_shape) == 5 // deciding whether it is an LSTM dataset

                    # save labels to the new dataset
                    new_dataset.create_dataset(output_keys[d][j_lbls], shape=lbls_shape, data=labels, dtype="uint8")
                # endfor i in range(0, len(input_keys[d])) // for every pair (data, labels) in the old dataset
            # endwith h5py.File(dataset, 'r') as old_dataset // open required dataset
        # endfor d, dataset in enumerate(input_dataset) // for each required dataset
    # endwith h5py.File(output_dataset, 'w') as new_dataset // create a new dataset


def get_img_paths(data, balance=None, lstm=False):
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
            paths = glob.glob(os.path.join(path, "*"))
            size = len(paths)
            if paths is []:
                continue

            if lstm:
                img_paths = list()
                for img_path in paths:
                    img_paths.append(glob.glob(os.path.join(img_path, "*")))
            else:
                img_paths = paths
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


def add_imgs(data, output, shape, normalize=True, verbose=True, lstm=False):

    num_data = len(data)
    t_last = time.time()

    # read all image paths
    for index, path in enumerate(data):
        # Print the status
        if verbose and (((time.time() - t_last) > PRINT_STATUS) or (index == 0)):
            # how many images was done
            print(f'[%6d/%6d] done' % (index, num_data))
            t_last = time.time()

        if lstm:
            for j, img_path in enumerate(path):
                # Check if the shape is correct
                if j >= shape[0]:
                    break

                # Load image and reshape it
                img = cv2.imread(img_path)
                img = cv2.resize(img, (shape[2], shape[1]), interpolation=cv2.INTER_CUBIC)

                # Add image into output_file
                output[index, j, ...] = img

                # Transform image between [0, 1]
                if normalize:
                    output[index, j, ...] /= 255.0

        else:
            # Load image and reshape it
            img = cv2.imread(path)
            img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

            # Add image into output_file
            output[index, ...] = img

            # Transform image between [0, 1]
            if normalize:
                output[index, ...] /= 255.0
    # endfor index, path in enumerate(data) // read all image paths


def unpack_dataset(input_dataset, output_dir, class_names):
    with h5py.File(input_dataset, 'r') as dataset:
        for i in range(0, len(class_names)):
            dir = output_dir + class_names[dataset[Y_TRAIN][i]]
            os.mkdir(dir)

        shape = dataset[X_TRAIN].shape
        for i in range(0, shape[0]):
            dir = output_dir + '\\%s\\%d' % (CLASS_NAMES_SH[dataset[Y_TRAIN][i, 0]], i)
            os.mkdir(dir)
            for j in range(0, shape[1]):
                cv2.imwrite(dir + '\\%d.jpg' % j, np.multiply(dataset[X_TRAIN][i, j, ...], 255))
