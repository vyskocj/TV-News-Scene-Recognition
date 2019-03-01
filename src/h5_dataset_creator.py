import h5py
import os, glob
import cv2
import random
import numpy as np


def create_dataset(train_data_paths, test_data_paths, output_filename, img_shape=None, shuffle=True,
                   cluster_names=None, balance=None):
    """ Creating an image dataset. Make sure that all pictures are of the same shape.

    Parameters
    ----------
    train_data_paths : list
        List of paths to train data. Each element of the list is equal to the directory of images.
        Each directory is equal to 1 cluster!
    test_data_paths : list
        Same as train_data_paths for test data.
    output_filename : str
        Name of output file (h5 format).
    img_shape : tuple, optional
        Reshaping data images, default is None.
    shuffle : bool, optional
        Set shuffle writing of images to a file, the default is True.
    cluster_names : list, optional
        Same list format as train or test_data_paths. Each element of the list is equal to the name of the cluster.
        If this parameter is not set, each cluster is indexed only by a number from 0 to the train_data_paths lenght.
    """

    ###########################################################################
    #############               Defining variables                #############
    ###########################################################################
    # List variables, info about images: _imgs - paths to images; _clusters - clusters index of all images
    train_imgs, train_clusters = list(), list()
    test_imgs, test_clusters = list(), list()
    # Number of train and test images
    num_train = 0
    num_test = 0

    ###########################################################################
    #############                      Start                      #############
    ###########################################################################
    # Print info
    print("Starting to create a dataset:")

    # If balance is not defined
    if balance == None:
        balance = [False] * len(train_data_paths)

    # Create list of images for train and test
    # Create list about clustering information
    # Get number of train and test images
    train_imgs, train_clusters, num_train = list_of_all_imgs(train_data_paths, balance)
    test_imgs, test_clusters, num_test = list_of_all_imgs(test_data_paths)

    # Define output shape if img_shape is None
    if img_shape == None:
        img = cv2.imread(train_imgs[0])
        img_shape = img.shape

    # Shuffle images
    if shuffle == True:
        train_imgs, train_clusters = shuffle_2lists(train_imgs, train_clusters)
        test_imgs, test_clusters = shuffle_2lists(test_imgs, test_clusters)

    # Define output dataset
    with h5py.File(output_filename, 'w') as output_file:
        output_file.create_dataset("train_imgs", shape=(num_train,) + img_shape, dtype="float32")
        output_file.create_dataset("test_imgs", shape=(num_test,) + img_shape, dtype="float32")

        # Adding info about classification of neural network
        output_file.create_dataset("train_clusters", shape=(num_train,), data=train_clusters, dtype="uint8")
        output_file.create_dataset("test_clusters", shape=(num_test,), data=test_clusters, dtype="uint8")

        # Adding name of classes to output file
        if cluster_names != None:
            string_dt = h5py.special_dtype(vlen=str)
            output_file.create_dataset("cluster_names", shape=(len(cluster_names),), dtype=string_dt)
            output_file['cluster_names'][:] = cluster_names

        # Adding images to output_file
        print("Adding train data:")
        add_imgs(train_imgs, output_file["train_imgs"], img_shape)

        print("- Adding test data:")
        add_imgs(test_imgs, output_file["test_imgs"], img_shape)

    # Print info
    print("Data was successfully converted.")


def list_of_imgs(source_imgs, extension="jpg"):
    """ Load all images from directory.

    Parameters
    ----------
    source_imgs : str
        Input path of directory where are images to load.
    extension : str
        Optional - images file format.

    Returns
    ----------
    paths : list of str
        List of paths of all images.
    """
    paths = glob.glob(os.path.join(source_imgs, "*." + extension))
    size = len(paths)
    return paths, size


def list_of_all_imgs(data, balance=None):
    data_paths = list()  # list for all images
    data_clusters = list()  # list of all classifications
    for index, path in enumerate(data):
        img_paths, size = list_of_imgs(path)  # get list of paths and count them

        # balance training images
        if balance != None and balance[index] == True:
            # get the directory path
            dir_path = os.path.abspath(os.path.join(os.path.dirname(img_paths[0])))

            # get the img names
            img_names = [name.split("\\")[-1] for name in img_paths]

            # create dictionary for images and throw all paths
            dict_imgs = dict()
            img_paths = list()

            for i, img in enumerate(img_names):
                # split the img name, ie.: scene1_frame5.jpg
                img_names[i] = img_names[i].split("_")

                # add img into dictionary (by type of scene)
                if img_names[i][0] in dict_imgs.keys():
                    dict_imgs[img_names[i][0]].append(img_names[i][1])
                else:
                    dict_imgs[img_names[i][0]] = [img_names[i][1], ]

            # Add the 1 random frame from each scene
            [img_paths.append(path + "\\" + img + "_" + random.choice(dict_imgs[img])) for img in dict_imgs.keys()]

            # set a new size
            size = len(img_paths)

        data_paths += img_paths  # connect 2 lists
        data_clusters += [index] * size  # output vector for training neural network
    return data_paths, data_clusters, len(data_paths)


def add_imgs(data, output, shape):
    num_data = len(data)
    for index, path in enumerate(data):
        # Print info
        if index % 50 == 0:
            print(" [" + str(index) + "/" + str(num_data) + "] image")

        # Loading image and reshape
        img = cv2.imread(path)
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

        # Add image into output_file and transforming it between [0, 1]
        output[index, ...] = img
        output[index, ...] /= 255.0


def shuffle_2lists(list1, list2):
    """ Shuffle 2 lists of dataset
    """
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    list1[:], list2[:] = zip(*combined)
    return list1, list2

if __name__ == "__main__":
    names = os.listdir("..\\data\\dataset\\train")

    tren = ["..\\data\\dataset\\train\\" + name for name in names]
    tst = ["..\\data\\dataset\\test\\" + name for name in names]

    blc = list()
    for i in tren:
        i = i.split("\\")[-1]
        if i in ["graphics", "indoor", "indoor studio of Czech TV", "outdoor country", "outdoor human made"]:
            blc.append(True)
        else:
            blc.append(False)

    create_dataset(tren, tst, "..\\data\\dataset.h5", img_shape=(180, 320, 3), cluster_names=names)
