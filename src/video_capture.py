import cv2
import math


# TODO: přehodit na lepší místo - do mainu
VIDEO_FILE = "../data/udalosti.ts"
OUTPUT_DIR = '../data/dataset_test_lstm/image'


# TODO: dokomentovat, název uloženého obrázku dle času
def video2img(file, output_directory, time_to_get_frame=0):
    # loading the video and getting frame rate
    cap = cv2.VideoCapture(file)

    # set the framerate to get image from video
    if time_to_get_frame == 0:
        framerate = 1
    else:
        framerate = cap.get(cv2.CAP_PROP_FPS) * time_to_get_frame
    print('framerate: ' + str(framerate) + ' [Hz]')

    # saving pictures from video
    index = 1
    while cap.isOpened():
        frame_id = cap.get(1)

        ret, frame = cap.read()
        if ret is not True:
            break
        if frame_id % math.floor(framerate) == 0:
            framename = output_directory + "0" * (4 - len(str(index))) + str(index) + ".jpg"
            index += 1
            cv2.imwrite(framename, frame)

            if index % 20 == 0:
                print("[" + str(index) + "] pictures saved")

    cap.release()


if __name__ == "__main__":
    video2img(VIDEO_FILE, OUTPUT_DIR, 0.2)
