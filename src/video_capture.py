import cv2
import math
import os

from src.const_spec import *
from ext import cut_detector as cd


# TODO: přehodit na lepší místo - do mainu
VIDEO_FILE = "../data/udalosti.ts"
OUTPUT_DIR = '../data/dataset_test_lstm'


def video2img(file, output_directory, resize=None, get_frame=0.0, unit='select', distinguish=False,
              min_frames=0, verbose=True):

    # Creating a new directory if not exists
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Warning that in the output directory may be some images
    if os.listdir(output_directory) != list():
        print('[W] Output directory is not empty! %s' % os.path.abspath(output_directory))

    # Loading the video and getting frame rate
    cap = cv2.VideoCapture(file)

    # Setting the framerate to get video image
    if get_frame == 0:
        get_frame = 1
    elif unit == 'time':
        get_frame = cap.get(cv2.CAP_PROP_FPS) * get_frame
    # endif get_frame == 0 // Setting the framerate to get video image

    # it should be an integer
    get_frame = math.floor(get_frame)

    # time increment
    dt = 1000.0 / cap.get(cv2.CAP_PROP_FPS)

    # Get total frames to save
    total_frames = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) / get_frame)

    # A video cutter is used to distinguish each scene
    cuts = list()
    if distinguish:
        if verbose:
            print('Using the Video Cutter')

        # Configuration for TV News
        config = cd.get_config('udalosti')

        # Score calculation to detect cuts
        scorings = cd.calculate_scorings(file, sizes=[config['size']])

        # Get cuts using TV News configuration and scorings
        cuts, means, maxs = cd.calculate_cuts(
            scorings['SAD_%d' % config['size']], config['neighbourhood_size'], config['neighbourhood_distance'],
            config['T1'], config['T2'], config['T3']
        )

        get_frame_list = list()
        cuts = [0] + cuts + [cap.get(cv2.CAP_PROP_FRAME_COUNT)]
        for i in range(0, len(cuts) - 1):
            if (cuts[i + 1] - cuts[i]) / get_frame < min_frames:
                save = int((cuts[i + 1] - cuts[i]) / min_frames)

                if save == 0:
                    save = 1
                get_frame_list.append(save)
            else:
                get_frame_list.append(get_frame)
        get_frame = get_frame_list
        cuts = cuts[1:-1]
    else:
        # The video cutter is not required
        cuts.append(math.inf)
        get_frame = [get_frame]
    # endif distinguish // A video cutter is used to distinguish each scene

    # Get number of cuts
    num_cuts = len(cuts)

    # Print information about image saving
    if verbose and min_frames in [0, 1]:
        if str(get_frame)[-1] == '1':
            termination = 'st'
        elif str(get_frame)[-1] == '2':
            termination = 'nd'
        elif str(get_frame)[-1] == '3':
            termination = 'rd'
        else:
            termination = 'th'

        print('Saving every ' + str(get_frame) + termination + ' frame.')
    # endif verbose // Print information about image saving

    # Saving images from the video
    saved = 0
    time = 0
    scene = 0
    scene_directory = ''
    while cap.isOpened():
        frame_id = int(cap.get(1))

        # Check that the video is not at the end
        ret, frame = cap.read()
        if ret is not True:
            break

        # Time to save frame
        if frame_id % get_frame[scene] == 0:
            # Use the video cutter information to divide images by scene type
            if distinguish and scene < num_cuts and frame_id >= cuts[scene]:
                # A new scene is detected - create a directory to store images
                scene += 1
                scene_directory = f'scene%04d' % scene
                if scene_directory not in os.listdir(output_directory):
                    os.mkdir(output_directory + '\\' + scene_directory)
            elif distinguish and time == 0:
                # Init directory for storing images when the video cutter is required
                scene_directory = f'scene%04d' % 0
                if scene_directory not in os.listdir(output_directory):
                    os.mkdir(output_directory + '\\' + scene_directory)

            # Get image name
            img_name = output_directory + '\\' + scene_directory + f'\\%08dms.jpg' % time

            # Resize image if required
            if resize is not None:
                frame = cv2.resize(frame, resize)

            # Save image to the output directory
            cv2.imwrite(img_name, frame)
            saved += 1

            if verbose and (saved % 50 == 0):
                if min_frames in [0, 1]:
                    print(f'[%5d/%5d] images saved' % (saved, total_frames))
                else:
                    print(f'[%5d/more than %5d] images saved' % (saved, total_frames))
        # endif frame_id % get_frame == 0 // Time to save frame

        time += dt
    # endwhile cap.isOpened() // Saving images from the video

    if verbose and (saved % 50 != 0):
        if min_frames in [0, 1]:
            print(f'[%5d/%5d] images saved' % (saved, total_frames))
        else:
            print('[the last] image saved')

    cap.release()


if __name__ == "__main__":
    video2img(VIDEO_FILE, '../data/dataset_test_lstm_2', resize=FRAME_SIZE, get_frame=0.2, unit='time', distinguish=True, min_frames=NUM_FRAMES)
    #video2img(VIDEO_FILE, '../data/dataset_test', resize=FRAME_SIZE, get_frame=1, unit='time', distinguish=False)
