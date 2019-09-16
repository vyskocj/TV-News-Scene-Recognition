#!/usr/bin/python
# -*- coding: utf-8 -*-
# $Id: cut_detector_annotation.py 84 2012-09-21 11:10:08Z pcampr $

import os
import numpy as np
import datetime


def parse_annotation_v1(lines):
    """
    lines like:

        # comment
        670 w 675 # another comment
        686
        722
        782
        860 d 901
        1025
        1668 d 1672
        1708 z 1720
        1813 p

    output: list of tuples, e.g. [(670,'w',675), (686,'',None), ...]
    """

    annotations = []

    for line in lines:
        comment_pos = line.find('#')

        if comment_pos>=0:
            line = line[:comment_pos]

        line = line.strip()

        if len(line) == 0:
            continue

        parts = line.split(' ')
        parts_nb = len(parts)

        assert parts_nb >= 1 and parts_nb <= 3

        # part 1
        f1 = parts[0]

        assert f1.isdigit()

        f1 = int(f1)

        # part 2
        if parts_nb > 1:
            type = parts[1]
        else:
            type = ''

        # part 3
        if parts_nb > 2:
            f2 = parts[2]

            assert f2.isdigit()

            f2 = int(f2)
        else:
            f2 = None

        annotations.append((f1, type, f2))

    return annotations


def get_annotation_file(dir, video_filename):
    return os.path.join(dir, os.path.basename(video_filename) + '.cuts.txt')


def read_for_video(dir, video_filename, length):
    """
    read annotation for video file from given directory
    """

    fn = get_annotation_file(dir, video_filename)

    return read(fn, length)


def read(fn, length=None):
    max_frame_nb = -1

    if not os.path.exists(fn):
        raise IOError('Annotation file not found: %s' % (fn))

    fr = open(fn)
    lines = fr.readlines()
    fr.close()

    if len(lines) > 0 and lines[0].strip() == '#cuts_v1':
        annotations = parse_annotation_v1(lines)
    else:
        raise Exception('Unknown annotation format')

    if length is None:
        # don't process other data, return only annotations
        return annotations

    cut_frames_truth = []
    cut_array_truth = np.zeros(length, dtype='uint32')

    for j, cut_truth_annotation in enumerate(annotations):
        if cut_truth_annotation[2] == None:
            idx = cut_truth_annotation[0]

            assert idx > max_frame_nb

            cut_frames_truth.append(idx)
            cut_array_truth[idx] = j + 1

            max_frame_nb = idx
        else:
            # transition cuts,
            # extend 1 frame left and right
            cut_from = cut_truth_annotation[0] - 1
            cut_to = cut_truth_annotation[2] + 1

            assert cut_from >= max_frame_nb
            assert cut_to > max_frame_nb
            assert cut_to - cut_from < 50 # transition duration - max. N frames

            cut_frames_truth += range(cut_from, cut_to + 1)
            cut_array_truth[cut_from: cut_to + 1] = j + 1

            max_frame_nb = cut_to

    cut_nb_truth = len(annotations)

    return (annotations, cut_frames_truth, cut_array_truth, cut_nb_truth)


def write(filename, cuts, comment = None):
    with open(filename, 'w') as fw:
        fw.write('#cuts_v1\n')
        fw.write('# written by $Id: cut_detector_annotation.py 84 2012-09-21 11:10:08Z pcampr $\n')
        fw.write('# time: %s\n' % (datetime.datetime.now()))
        fw.write('# cuts: %d\n' % (len(cuts)))

        if comment is not None:
            fw.write('# comment:\n')

            for line in comment.splitlines():
                fw.write('#   %s\n' % (line))

        for cut in cuts:
            fw.write('%d\n' % (cut))
