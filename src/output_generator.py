import keras
import h5py
import os
import cv2
import numpy as np

IMG_SHAPE = (180, 320)


def generate_traindata_to_h5(input_directory, output_h5_file, model, num_images_for_classify=50):
    with h5py.File(output_h5_file, 'w') as output_file:

        img_list = list()
        for img_name in os.listdir(input_directory):
            img = cv2.imread(input_directory + '\\' + img_name)
            img = cv2.resize(img, (IMG_SHAPE[1], IMG_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
            img = np.array(img)
            img = img.astype("float32")
            img = np.divide(img, 255.0)

            img_list.append(img)

            if len(img_list) == num_images_for_classify:
                break

        img_list = np.array(img_list)
        pred = model.predict(img_list)

        output_file.create_dataset("train_imgs", shape=(1, len(pred), len(pred[0])), dtype="float32")
        output_file.create_dataset("train_clss", shape=(1, ), dtype="uint8")

        output_file["train_imgs"][0, ...] = pred
        output_file["train_clss"][0] = 1

        print(output_file)


if __name__ == "__main__":
    model = keras.models.load_model('..\\models\\part1.h5')

    generate_traindata_to_h5('../data/dataset_test', '..\\data\\test_z_vygenerovanych_50_vystupu.h5', model)
