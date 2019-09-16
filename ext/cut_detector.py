#!/usr/bin/python
# -*- coding: utf-8 -*-
# $Id: cut_detector.py 95 2012-09-26 19:35:52Z pcampr $

"""
Method reimplemented from:
  http://www.kky.zcu.cz/cs/publications/MatousekJ_2012_ImprovingAutomatic
"""

import os
import cv2
import numpy as np
import json
from ext import cut_detector_annotation


def get_config(id):
    """
    Read config file from 'config' subdirectory
    """
    fn = os.path.join( os.path.dirname(__file__), 'config', id+'.json')

    if not os.path.exists(fn):
        raise Exception('Config file not found: %s' % (fn))

    with open(fn) as fr:
        config = json.load(fr)

    config['_file'] = fn

    return config


def resize_keep_ratio(img, size, type='max'):
    """
    Resize image and keep aspect ratio by setting only width or height, or maximum of both
    """
    h = img.shape[0]
    w = img.shape[1]

    if type == 'max':
        if img.shape[0] > img.shape[1]:
            # image is tall - "I" shape
            type = 'height'
        else:
            # image is wide - "-" shape
            type = 'width'

    if type == 'width':
        set_width = size
        ratio = 1.0 * set_width / w
        set_height = int(ratio * h)
    elif type == 'height':
        set_height = size
        ratio = 1.0 * set_height / h
        set_width = int(ratio * w)
    else:
        raise Exception('Wrong type')

    return cv2.resize(img, (set_width, set_height))


def calculate_scorings_cached(scoring_dir, fn, sizes=[150]):
    scorings_cached = {}
    sizes_not_cached = []
    frame_nb = None

    if not os.path.isdir(scoring_dir):
        raise Exception('Directory not found: %s' % (scoring_dir))

    # use scoring_dir as a cache directory,
    cache_file_mask = os.path.join(scoring_dir, os.path.basename(fn) + '.scoring_%s.txt')

    # load from cache
    for size in sizes:
        id = 'SAD_%d' % (size)
        cache_file = cache_file_mask % (id)

        if os.path.exists(cache_file):
            scorings_cached[id] = np.loadtxt(cache_file).astype('float16')
            frame_nb = len(scorings_cached[id])
            print('    (using cached result %s)' % (id))
        else:
            sizes_not_cached.append(size)

    if len(scorings_cached) == len(sizes):
        # all sizes are cached, return result now
        print('    (all results are cached)')
        return (scorings_cached, frame_nb)

    # calculate scorings not available in cache
    scorings = calculate_scorings(fn, sizes=sizes)

    # save new results to cache
    for id, v in scorings.items():
        cache_file = cache_file_mask % (id)
        np.savetxt(cache_file, v, fmt='%.4e', delimiter='\n')
        frame_nb = len(v)

    scorings.update(scorings_cached)

    return (scorings, frame_nb)


def calculate_scorings(fn, max_frames=None, sizes=[150]):
    if not os.path.exists(fn):
        raise Exception('File not found: %s' % (fn))

    # init video
    cap = cv2.VideoCapture(fn)

    if cap is None or not cap.isOpened():
        raise Exception('Unable to open video source: %s' % (fn))

    # init variables
    i = -1
    imgs = {}
    scorings = {}

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    for size in sizes:
        id = 'SAD_%d' % (size)
        scorings[id] = [0]
        imgs[size] = []

    # process frames
    while max_frames is None or i < max_frames:
        i += 1

        # prepare image
        ret, img = cap.read()

        if not ret:
            break

        # compute scorings for each size
        for size in sizes:
            id = 'SAD_%d' % (size)

            img_resized = resize_keep_ratio(img, size, type='height')
            imgs[size].append(img_resized)

            if len(imgs[size]) > 5:
                imgs[size][-5] = None # remember only last N frames

            # scoring for current frame
            if len(imgs[size]) > 1:
                # SAD - sum of absolute differences, normalize sum by size
                d = cv2.absdiff(imgs[size][-2], imgs[size][-1]) # .astype('float32') # / 255
                sum = np.sum(d) / np.size(d)

                scorings[id].append(sum)

        # output
        if i%1000==0 or i==max_frames:
            print('    frame %d/%d %s\r' % (i, total_frames, ' '.join(['%s=%4.1f' % (k, v[-1]) for k, v in scorings.items()])[:40])),

    print()

    cap.release()

    for size in sizes:
        id = 'SAD_%d' % (size)
        scorings[id] = np.array(scorings[id], dtype='float16')

    return scorings


def calculate_cuts(scoring, neighbourhood_size=16, neighbourhood_distance=1, T1=9, T2=1, T3=0.45, means=None, maxs=None):
    """
    Original proposed parameters:
      T1=12.54, T2=0.675, T3=0.529
      neighbourhood_distance=0
      neighbourhood_size=7

    New parameters from cut_detector_train.py:
      T1=9, T2=1, T3=0.45
      neighbourhood_distance=1
      neighbourhood_size=16

                                              current frame
                                                    |
    <+++LLLLLLLLLLLLLLLLLLLL++++++++++++++++++++++++|++++++++++++++++++++++++RRRRRRRRRRRRRRRRRRRR+++> time
        [neighbourhood_size][neighbourhood_distance]|[neighbourhood_distance][neighbourhood_size]
         left neigbourhood                          |                        right neighbourhood

    """

    cuts = []

    neighbourhood_size = int(neighbourhood_size)
    neighbourhood_distance = int(neighbourhood_distance)

    scoring_len = len(scoring)

    # apply rule 1
    idxs = np.argwhere(scoring > T1).ravel()

    for i in idxs:
        score_in_i = scoring[i]

        L = int(round(i - neighbourhood_distance - neighbourhood_size / 2))
        R = int(round(i + neighbourhood_distance + neighbourhood_size / 2))

        if L < 0 or R >= scoring_len:
            continue

        # apply rule 3
        if means is None:
            means = np.convolve(scoring, np.ones(neighbourhood_size) / neighbourhood_size, 'same')

        if (means[L] + means[R]) / 2 > T3 * score_in_i:
            continue

        # apply rule 2
        if maxs is None:
            maxs = np.zeros_like(means)

            for j in range(neighbourhood_size // 2, len(scoring) - neighbourhood_size // 2):
                maxs[j] = np.max(scoring[j - neighbourhood_size // 2:j + neighbourhood_size // 2 + 1])

        thresh = T2 * score_in_i

        if maxs[L] > thresh or maxs[R] > thresh:
            continue

        cuts.append(i)

    return (cuts, means, maxs)


if __name__ == '__main__':
    import argparse

    # parse commandline
    parser = argparse.ArgumentParser(description='Analyze video files, detect cuts')
    parser.add_argument('-D', '--debug', action='store_true', help='enable debug output')
    parser.add_argument('-Y', '--overwrite', action='store_true', help='overwrite existing files')
    parser.add_argument('-a', '--annotation_dir', help='Annotation directory used to evaluate results')
    parser.add_argument('config_file', help='Configuration file or ID')
    parser.add_argument('dir', help='Output and scoring cache directory')
    parser.add_argument('videos', type=str, nargs='+', default=[], help='Videos to process')

    args = parser.parse_args()

    if not len(args.videos):
        parser.print_usage()

    config = get_config(args.config_file)

    sizes = [config['size']]

    videos_nb = len(args.videos)

    for i, fn in enumerate(args.videos):
        print('Processing video (%d of %d) %s' % (i + 1, videos_nb, fn))

        result_fn = os.path.join(args.dir, os.path.basename(fn) + '.cuts.txt')

        if os.path.exists(result_fn):
            if args.overwrite:
                print('    output file exists, overwriting')
                os.remove(result_fn)
            else:
                if not args.annotation_dir:
                    print('    output file exists, skipping')
                    continue
                else:
                    print('    output file exists, but evaluating with annotation')
        else:
            # create empty file as a lock
            open(result_fn, 'w').close()

        scorings, frame_nb = calculate_scorings_cached(args.dir, fn, sizes)

        for size in sizes:
            id = 'SAD_%d' % (size)
            cuts, means, maxs = calculate_cuts(scorings[id], config['neighbourhood_size'], config['neighbourhood_distance'],
                config['T1'], config['T2'], config['T3'])
            print('    %s cuts: %d' % (id, len(cuts)))

        if args.debug:
            import matplotlib.pyplot as plt

            for size in sizes:
                id = 'SAD_%d' % (size)
                plt.plot(scorings[id], label=id)

            plt.title('SAD')
            plt.show()

        # write result
        cut_detector_annotation.write(result_fn, cuts, comment='version: $Id: cut_detector.py 95 2012-09-26 19:35:52Z pcampr $\nconfig: '+str(config))

        if args.annotation_dir:
            import cut_detector_train

            print('    evaluating results:')

            try:
                annotation = cut_detector_annotation.read_for_video(args.annotation_dir, fn, frame_nb)
            except IOError as e:
                print('        Cannot read annotation: %s' % (e.strerror))
                continue

            F, FP_cuts, TP_cuts, FN_cuts = cut_detector_train.F_score(annotation[1], annotation[2], annotation[3], cuts, F_beta=2, extended_results=True)

            print('        F2 score: %.4f' % (F))
            print('        TP: %d' % (len(TP_cuts)))
            print('        FP: %d' % (len(FP_cuts)))
            print('        FN: %d' % (len(FN_cuts)))

            diff_fn = result_fn.replace('.txt', '.annotation_diff.txt')

            with open(diff_fn, 'wt') as fw:
                fw.write('# F2 score: %f\n' % (F))

                fw.write('# FN (wrongly not-detected cuts):\n')
                for cut in FN_cuts:
                    if type(cut) == tuple:
                        fw.write('%d ? %d\n' % (cut[0], cut[1]))
                    else:
                        fw.write('%d\n' % (cut))

                fw.write('# FP (wrongly detected cuts):\n')
                for cut in FP_cuts:
                    fw.write('%d\n' % (cut))

                fw.write('# TP (correctly detected cuts):\n')
                for cut in TP_cuts:
                    fw.write('%d\n' % (cut))
