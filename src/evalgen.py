import os
import shutil
import cv2
import json
import re
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from vis.visualization import visualize_cam

from src.const_spec import *

E_TEMPLATE_CLASSES = '[E] Pass the parameter to the template function: ' \
                     '"classes" - a list of neural network output names.'
E_TEMPLATE_VIEW = '[E] Pass the parameter to the template function: ' \
                  '"view" - path to the image / gif that represents prediction.'
E_TEMPLATE_OUTPUT_VECTOR = '[E] Pass the parameter to the template function: ' \
                           '"output_vector" - the output of predict method.'
E_TEMPLATE_PATH_OR_GHOST = '[E] Pass the parameter to the template function: ' \
                           '"path" or "ghost" - list of image paths.'
E_TEMPLATE_GRAD_CAM = '[E] Pass the parameter to the template function: ' \
                      '"heat_file" - path to the image that represents the heat map of prediction.'
E_REPLACE_TAG_NOT_FOUND = '[E] The tag not found in template.'
W_TEMPLATE_VALID_MATRIX_MISSING = '[W] Validation matrix is missing, the summary table is not generated.'
W_CREATE_HNTML_GC_NOT_SUP = '[W] The grad-cam is not supported for LSTM network.'


def get_time(filename):
    t = re.search(r'(\d+)ms', filename)
    if t is not None:
        t = f'%3d:%02d:%03d [min:s:ms]' % \
               (int(t[1]) / 60_000, (int(t[1]) / 1000) % 60, (int(t[1]) % 1000))
    else:
        t = ''

    return t


def template(part, file, **kwargs):
    if part == 'evaluation.prepare_to_predict':
        if 'classes' not in kwargs.keys():
            raise Exception(E_TEMPLATE_CLASSES)

        if 'validation' in kwargs.keys() and kwargs['validation'] is True:
            width = 80 / (len(kwargs['classes']) + 2)
            true_label = f'<th style="width:%.2f%%">True label</th>' % width
            if 'matrix' in kwargs.keys():
                if 'acc_loss' not in kwargs.keys():
                    kwargs['acc_loss'] = None

                summary = get_html_validation(kwargs['matrix'], kwargs['classes'], normalize=False, spaces='    ',
                                              acc_loss=kwargs['acc_loss'])[4:]
            else:
                print(W_TEMPLATE_VALID_MATRIX_MISSING)
                summary = ''
        else:
            width = 80 / (len(kwargs['classes']) + 1)
            true_label = ''
            summary = '<a href="./timeline.svg" target="_blank"><img src="./timeline.svg" width="100%" /></a>'

        replace(string=str(kwargs['classes']), old_tag='classArray', file=file)

        nav_bar = ''
        class_table = ''
        for c in kwargs['classes']:
            nav_bar += '        <a href="javascript:void(0)" id="%s" onclick="showById(\'%s\')">%s</a>\n' % (c, c, c)
            class_table += (f'            <th style="width:%.2f%%">' % width) + c + '</th>\n'

        replace(string=nav_bar[8:-1], old_tag='nav_bar', file=file)             # 8  -> number of spaces, -1 -> '\n'
        replace(string=summary, old_tag='summary', file=file)
        replace(string=class_table[12:-1], old_tag='class_table', file=file)    # 12 -> number of spaces, -1 -> '\n'
        replace(string=true_label, old_tag='true_label', file=file)
        replace(string='', old_tag='evaluation_predict', file=file)

    elif part == 'evaluation.predict':
        if 'view' not in kwargs.keys():
            raise Exception(E_TEMPLATE_VIEW)

        elif 'output_vector' not in kwargs.keys():
            raise Exception(E_TEMPLATE_OUTPUT_VECTOR)

        elif 'classes' not in kwargs.keys():
            raise Exception(E_TEMPLATE_CLASSES)

        elif 'heat_file' not in kwargs.keys() and 'grad_cam' in kwargs.keys() and kwargs['grad_cam'] is True:
            raise Exception(E_TEMPLATE_GRAD_CAM)

        view = '                '
        if 'grad_cam' in kwargs.keys() and kwargs['grad_cam'] is True:
            view += '<span class="img">'

        if 'gif' not in kwargs.keys() or ('gif' in kwargs.keys() and kwargs['gif'] is None):
            view += '<img src="' + kwargs['view'] + '" title="' + get_time(kwargs['view']) + '" />'
            if 'grad_cam' in kwargs.keys() and kwargs['grad_cam'] is True:
                view += '</span>\n'
                view += '                <span class="focus">'
                view += '<img src="' + kwargs['heat_file'] + '" title="' + get_time(kwargs['heat_file']) + '" /></span>'
            view += '\n'
        else:
            # LSTM with
            view += '<img src="' + kwargs['gif'] + '" title="' + get_time(kwargs['gif']) + '" />\n'

        if 'predict_file' in kwargs.keys() and kwargs['predict_file'] != '':
            # LSTM - Grad CAM is not supported - so do not care about line alignment for view variable!
            view = '                <a href="./predicts/' + kwargs['predict_file'] + '.html" target="_blank">\n' \
                   '    ' + view + \
                   '                </a>\n'

        output_vector = ''
        for i, decision in enumerate(kwargs['output_vector']):
            output_vector += '            <td title="' + kwargs['classes'][i] + '">' + \
                             ('<b>' if decision == max(kwargs['output_vector']) else '') + (f'%.6f' % decision) + \
                             ('</b>' if decision == max(kwargs['output_vector']) else '') + '</td>\n'

        if 'true_label' in kwargs.keys():
            tr_class = kwargs['classes'][np.argmax(kwargs['true_label'])]
            true_label = '            <td title="True label">' + tr_class + '</td>\n'
        else:
            tr_class = kwargs['classes'][np.argmax(kwargs['output_vector'])]
            true_label = ''

        string = (
            '        <tr class="' + tr_class + '">\n'
            '            <td class="td-img">\n'
            + view +
            '            </td>\n'
            + output_vector +
            '            <td title="Predicted">' + kwargs['classes'][np.argmax(kwargs['output_vector'])] + '</td>\n'
            + true_label +
            '        </tr>\n'
        )

        insert(string, file)

    elif part == 'predict.img_list':
        if 'path' not in kwargs.keys() and 'ghost' not in kwargs.keys():
            raise Exception(E_TEMPLATE_PATH_OR_GHOST)

        if 'back_step' not in kwargs.keys():
            kwargs['back_step'] = ''

        img_paths = ''
        if 'path' in kwargs.keys():
            for path in kwargs['path']:
                src = kwargs['back_step'] + path
                img_paths += '    <img src="' + src + '" style="max-width:400px"' \
                             ' title="' + get_time(path) + '" />\n'

        ghost_paths = ''
        if 'ghost' in kwargs.keys():
            for path in kwargs['ghost']:
                src = kwargs['back_step'] + path
                ghost_paths += '    <img src="' + src + '" style="max-width:400px; opacity:0.3"' \
                               ' title="' + get_time(path) + '" />\n'

        file.write(
            '<html>\n'
            '  <body>\n'
            + img_paths + ghost_paths +
            '  </body>\n'
            '</html>'
        )


def get_back_step(input_path, output_path):
    input_step = input_path.split('\\')
    output_step = output_path.split('\\')

    output_step.remove('..')

    return '../' * (len(output_step) - input_step.count('..'))  # multiplying by a negative number does not matter


def insert(string, file):
    seek = file.tell()             # The actual position of the pointer
    rest_lines = file.readlines()  # The rest of the lines from file
    file.seek(seek)

    file.write(string)
    seek = file.tell()
    file.writelines(rest_lines)
    file.seek(seek)


def replace(string, old_tag, file):
    last_seek = file.tell()                # pointer to the line from last cycle
    num_rem_lines = len(file.readlines())  # number of remaining lines
    file.seek(last_seek)

    # Finding a tag that needs to be replaced with a new string
    line = file.readline()  # current reading line
    while num_rem_lines > 0:
        # Looking for the tag
        regex = re.search(r'<\?\s*%s\s*\?>' % old_tag, line)
        if regex is not None:
            # New line without the tag
            line = line[:regex.regs[0][0]] + string + line[regex.regs[0][1]:]

            # Write new line
            rest_lines = file.readlines()   # The rest of the lines from file
            file.seek(last_seek)            # Move pointer to the beginning of the line
            file.write(line)

            # Write the rest of the lines to the file and move seek to the last position
            file.writelines(rest_lines)
            file.truncate(file.tell())
            file.seek(last_seek)

            return
        # endif regex is not None // Looking for the tag

        # next step
        last_seek = file.tell()
        line = file.readline()
        num_rem_lines -= 1
    # endwhile line is not None // Finding a tag that needs to be replaced with a new string

    raise Exception(E_REPLACE_TAG_NOT_FOUND)


def create_html(model, classes, input_path, output_path, portable=True, grad_cam=False, adaptive_strides=True,
                verbose=True):
    if verbose:
        print('[I] Creating the HTML file with predictions.')

    # Get input shape of model
    input_shape = model.get_input_shape_at(0)

    # Get number of input frames needed for single decision
    if len(input_shape) == 5:
        # Model type is LSTM
        num_frames = input_shape[1]
        if grad_cam is True:
            print(W_CREATE_HNTML_GC_NOT_SUP)
            grad_cam = False
    else:
        # Model is Time-Independent
        num_frames = 1
    # endif  len(input_shape) == 5 // Get number of input frames needed for single decision

    # Get image shape
    img_shape = input_shape[-3:]

    # LSTM Network - create directory for the html files with the images that were used to predict
    if num_frames > 1:
        if not os.path.exists(os.path.join(output_path, 'predicts')):
            os.mkdir(os.path.join(output_path, 'predicts'))
        if not os.path.exists(os.path.join(output_path, 'gifs')):
            os.mkdir(os.path.join(output_path, 'gifs'))

    # The images will be copied to the HTML file
    if portable and not os.path.exists(os.path.join(output_path, 'imgs')):
        os.mkdir(os.path.join(output_path, 'imgs'))

    # The output of Grad CAM
    if grad_cam and not os.path.exists(os.path.join(output_path, 'focuses')):
        os.mkdir(os.path.join(output_path, 'focuses'))

    # Copy template and rename it
    shutil.copy(os.path.join(os.path.dirname(__file__), '..', EVALTEMP), os.path.join(output_path, 'evaluation.html'))

    # Write the evaluation to the html file
    with open(os.path.join(output_path, 'evaluation.html'), 'r+') as html_file:
        # Load the template and save it to the html file
        template('evaluation.prepare_to_predict', html_file, classes=classes)

        # Compute how many back steps it takes from html file to the images
        if portable:
            back_step = './'
        else:
            back_step = get_back_step(input_path, output_path)

        # Lists of input images that going to be passed to the Neural Network
        input_imgs = {
            'path': list(),     # path to the image
            'imgs': list(),     # image that is passed thought network
            'ghost': list(),    # path to the image that was used to complete the required number of input images
            'len': 0
        }

        # The timeline of decisions
        decisions = {
            'class': list(),
            'time': list()
        }
        last_decision = None    # Last classification
        last_img_name = ''      # Last image name -> it is named after the time in milliseconds it was taken

        # Variables for LSTM
        file_index = 0          # Index of the html file with all frames that were used to predict
        predict_file = ''       # The name of the html file with all frames that were used to predict
        pivot = 0               # Pivot is used to identify how many images can be used to predict
        read_rem_imgs = False   # Read the remaining images from the directory
        num_dirs_in_path = 0    # Counter that rises with each detection that a path leads to a directory
        num_total_files = len(os.listdir(input_path))  # The total number of files in the input path

        # Variable for Grad CAM
        heat_file = ''          # The name of the img file that shows what the Neural Network sees

        # Read data from the given directory
        t_last = time.time()
        for num_processed, file_name in enumerate(os.listdir(input_path)):
            path = os.path.join(input_path, file_name)

            # Check if the file is directory
            if os.path.isdir(path):
                # The directory represents one scene
                pivot = 0                                # Pivot is changed only when directory has changed
                directory = os.listdir(path)             # List files in directory
                num_imgs_in_dir = len(os.listdir(path))  # Number of images in the directory
                num_dirs_in_path += 1                    # Increment number of directories in path
            else:
                # The file is not a directory but an image
                path = input_path                                             # Change the path as input_path
                directory = [file_name]                                       # Pass image name thought for cycle
                num_imgs_in_dir = num_total_files - num_dirs_in_path  # Compute how many imgs can be in the dir
            # endif os.path.isdir(path) // Check if the file is directory

            if verbose and (time.time() - t_last) > PRINT_STATUS:
                # how many dirs / images was done
                print(f'[%3d %%] done' % (num_processed * 100 / num_total_files))
                t_last = time.time()

            for img_name in directory:
                # Load image and prepare it for model prediction
                path_to_img = os.path.join(path, img_name)
                img = cv2.imread(path_to_img)
                img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
                if portable:
                    path_to_img = os.path.join('imgs', img_name)
                    cv2.imwrite(os.path.join(output_path, path_to_img), img)
                img = img.astype("float32")
                img = np.divide(img, 255.0)

                # Lists length increment, save the image and its path
                input_imgs['path'].append(path_to_img)
                input_imgs['imgs'].append(img)
                input_imgs['len'] += 1

                # Check if images can be passed to the network
                if input_imgs['len'] < num_frames and \
                        (input_imgs['len'] < (num_imgs_in_dir - pivot * num_frames) or read_rem_imgs):
                    # There are still some images that can be loaded
                    # Save last image name - it represents when the image was taken - and continue
                    last_img_name = img_name
                    continue

                # Copy the last image into the remaining empty fields of the array. This is needed when fewer images
                # are available than they are needed to calculate the prediction.
                if num_frames > 1:
                    cop = int(input_imgs['len'] / 2)  # copy image from the middle
                    for i in range(0, num_frames - input_imgs['len']):
                        input_imgs['imgs'].append(input_imgs['imgs'][cop])
                        input_imgs['ghost'].append(input_imgs['path'][cop])

                # The network prediction
                if num_frames > 1:
                    # LSTM network is used
                    network_say = model.predict(np.array([input_imgs['imgs']]))[0]
                else:
                    # Time-Independent network
                    network_say = model.predict(np.array(input_imgs['imgs']))[0]

                # Save decision of Neural Network if it is needed
                decision = np.argmax(network_say)
                if decision != last_decision:
                    # Save decision of last image
                    if last_decision is not None:
                        decisions['class'].append(last_decision)
                        decisions['time'].append(int(re.search(r'(\d+)ms', last_img_name)[1]))
                    else:
                        decisions['class'].append(decision)
                        decisions['time'].append(0)

                    # Save decision of current image
                    decisions['class'].append(decision)
                    decisions['time'].append(int(re.search(r'(\d+)ms', img_name)[1]))

                    # Save this decision as recent
                    last_decision = decision
                # endif decision != last_decision // Save decision of Neural Network if it is needed

                gifname = None
                if num_frames > 1:
                    # Create gif
                    gif = []
                    for i in input_imgs['path']:
                        gif.append(Image.open(os.path.join(output_path, i)))

                    ms = 100
                    crt_gif = False
                    if len(input_imgs['path']) >= 2:
                        # Get actual framerate
                        ms_1 = re.search(r'(\d*)ms', input_imgs['path'][0])
                        ms_2 = re.search(r'(\d*)ms', input_imgs['path'][1])

                        if ms_1 is not None and ms_2 is not None:
                            ms = int(ms_2[1]) - int(ms_1[1])
                            crt_gif = True
                        else:
                            crt_gif = False
                            gifname = None

                    if crt_gif:
                        gifname = os.path.join('gifs', re.search(r'(\d*ms)\.', img_name)[1] + '.gif')
                        gif[0].save(os.path.join(output_path, gifname),
                                    save_all=True, append_images=gif[1:], duration=ms, loop=0)
                        gifname = './' + gifname

                    predict_file = f'%06.d' % file_index
                    with open(os.path.join(output_path, 'predicts', predict_file + '.html'), 'w') as p_file:
                        template('predict.img_list', p_file, path=input_imgs['path'], ghost=input_imgs['ghost'],
                                 back_step=(back_step + '../'))
                    file_index += 1
                elif grad_cam:
                    # What the Neural Network see - it is not supported for LSTM network
                    heat_map = visualize_cam(model, -1, decision, img)
                    heat_file = os.path.join(back_step, 'focuses', img_name)
                    cv2.imwrite(os.path.join(output_path, 'focuses', img_name), heat_map)
                # Saving images to the directory

                template('evaluation.predict', html_file, predict_file=predict_file, output_vector=network_say,
                         classes=classes, view=(back_step + input_imgs['path'][0]),
                         grad_cam=grad_cam, heat_file=heat_file, gif=gifname)

                # Prepare to next step
                pivot += 1

                # Removing images depending on the remaining images in the directory
                if adaptive_strides and num_frames > 1:
                    remaining = num_imgs_in_dir - pivot * num_frames

                    if remaining <= 0 or remaining >= num_frames:
                        input_imgs = {
                            'path': list(),
                            'imgs': list(),
                            'ghost': list(),
                            'len': 0
                        }
                        read_rem_imgs = False
                    else:
                        input_imgs = {
                            'path': input_imgs['path'][remaining:],
                            'imgs': input_imgs['imgs'][remaining:],
                            'ghost': list(),
                            'len': num_frames - remaining
                        }
                        read_rem_imgs = True
                else:
                    input_imgs = {
                        'path': list(),
                        'imgs': list(),
                        'ghost': list(),
                        'len': 0
                    }
                    read_rem_imgs = False

                # Change the last image name if model is Time-Independent
                if num_frames == 1:
                    last_img_name = img_name

        with open(os.path.join(output_path, 'decisions.json'), 'w') as json_file:
            decisions['class'] = np.array(decisions['class'], dtype='int8')
            decisions['time'] = np.array(decisions['time'], dtype='int64')

            decisions['class'] = decisions['class'].astype('int8').tolist()
            decisions['time'] = decisions['time'].astype('int64').tolist()

            json.dump(decisions, json_file)

        plt.figure(figsize=(20, 5))

        t = [x / 60_000 for x in decisions['time']]

        plt.plot(t, decisions['class'])
        plt.xticks(np.arange(min(t), max(t), 5))
        plt.yticks(np.arange(len(classes)), classes)

        # Show the major grid lines with dark grey lines
        plt.grid(b=True, which='major', color='#7F7F7F', linestyle='-')

        # Show the minor grid lines with bright grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#C0C0C0', linestyle='-', alpha=0.3)

        plt.xlim((min(t), max(t)))

        plt.title('Timeline of decisions')
        plt.ylabel('Classification')
        plt.xlabel('Time [min]')
        plt.legend(['Decision'], loc='upper left')

        plt.savefig(os.path.join(output_path, 'timeline.svg'))

    if verbose:
        print('[I] The HTML file with predictions was successfully created.')


def get_confusion_matrix(model, data, verbose=True):
    if verbose:
        print('[I] Generating the Confusion matrix.')

    x = data[0]
    y = data[1]

    if len(x.shape) == 5:
        batch_size = BATCH_SIZE_TD
    else:
        batch_size = BATCH_SIZE

    num_data = len(y)
    num_classes = len(y[0])
    matrix = np.zeros((num_classes, num_classes), dtype='uint16')
    predict = model.predict(x, batch_size=batch_size)

    wrong_pred = {
        'position': list(),
        'prediction': list()
    }
    for i in range(0, num_data):
        network_y = np.argmax(predict[i])
        supervisor_y = np.argmax(y[i])
        matrix[supervisor_y, network_y] += 1
        if network_y != supervisor_y:
            wrong_pred['position'].append(i)
            wrong_pred['prediction'].append(predict[i])

    if verbose:
        print('[I] The Confusion matrix was successfully generated.')

    return matrix, wrong_pred


def create_tex_validation(matrix, class_names, output_path, normalize=True, label=None, verbose=True):
    if verbose:
        print('[I] Creating the Confusion table for LaTeX.')

    if label is None:
        label = 'my_label'

    with open(os.path.join(output_path, 'confusion_matrix.tex'), 'w') as tex_file:
        num_classes = len(class_names)

        tex_file.write(
            '% This table uses the packages below\n'
            '% \\usepackage{graphicx}\n'
            '% \\usepackage[table, xcdraw]{xcolor}\n'
            '% \\usepackage{multirow}\n'
            '% \\usepackage{booktabs}\n'
            '\n'
            '\\begin{table}[h]\n'
            '    \\centering\n'
            '    \\resizebox{\\textwidth}{!}{%\n'
            '    \\begin{tabular}{ll|' + ('c' * num_classes) + '}\n'
            '       \\multicolumn{1}{l}{} & \\multicolumn{' + str(num_classes + 1) + '}{c}{Predicted label}\\\\\n'
            '        &'
        )

        [tex_file.write(' & \\rotatebox[origin=l]{90}{' + class_name + '}') for class_name in class_names]

        tex_file.write('\\\\ \\cmidrule(l){2-' + str(num_classes + 2) + '}\n')

        mat_sum = 1
        for c in range(0, num_classes):

            if c < num_classes - 1:
                tex_file.write('        & ')
            else:
                tex_file.write('        \\multirow{-' + str(num_classes + 1) + '}{*}{\\rotatebox{90}{True label}} & ')

            tex_file.write('\\multicolumn{1}{l|}{' + class_names[c] + '}')

            if normalize is True:
                mat_sum = matrix[c].sum()

            den = matrix[c].max()

            [tex_file.write(' & \\cellcolor[gray]{' + str(1 - matrix[c, i] / 2 / den) + '}'
                            + (f'%.3f' % (matrix[c, i] / mat_sum)))
             for i in range(0, num_classes)]

            tex_file.write('\\\\\n')

        tex_file.write(
            '     \\end{tabular}}\n'
            '     \\caption{' + ('Normalized c' if normalize else 'C') + 'onfusion matrix}\n'
            '     \\label{tab:' + label + '}\n'
            '\\end{table}\n'
        )

    if verbose:
        print('[I] The LaTeX Confusion table was successfully created.')


def get_html_validation(matrix, class_names, normalize=True, spaces='', acc_loss=None):
    width = int(100 / (len(class_names) + 2))
    first_line = spaces + '        <td width="' + str(width) + '%"></td>\n'

    for label in class_names:
        first_line += spaces + '        <td width="' + str(width) + '%">' + label + '</td>\n'

    table = ''
    mat_sum = 1
    correct = 0
    for i, line in enumerate(matrix):
        table += spaces + '    <tr>\n'
        table += spaces + '        <td>' + class_names[i] + '</td>\n'
        if normalize is True:
            mat_sum = matrix[i].sum()
            if mat_sum == 0:
                # there is no data in the current row
                mat_sum = 1

        den = matrix[i].max()
        if den == 0:
            # there is no data in the current row
            den = 1

        for j, val in enumerate(line):
            correct += 0 if i != j else val

            color = int((1 - val / 2 / den) * 255)
            table += (
                    spaces + '        <td style="background:rgb' + str((color, color, color)) + '">' +
                    str(f'%.3f' % (val / mat_sum) if normalize else int(val)) + '</td>\n'
            )

        table += spaces + '    </tr>\n'

    # link to acc and loss graphs
    if acc_loss is not None:
        fit_info = spaces + '    (<a href="%s">Accuracy</a>, <a href="%s">Loss</a>)\n' % (acc_loss[0], acc_loss[1])
    else:
        fit_info = ''

    return (
        spaces + '<table width="100%">\n' +
        spaces + '    <caption>Predicted label</caption>\n' +
        spaces + '    <tr>\n' +
        spaces + '        <td rowspan="' + str(len(class_names) + 1) + '">True label</td>\n'
        + first_line +
        spaces + '    </tr>\n'
        + table +
        spaces + '</table>\n' +
        spaces + '<br />\n' +
        spaces + '<div>\n' +
        spaces + '    Final accuracy: %.2f %%\n' % (correct / matrix.sum() * 100)
        + fit_info +
        spaces + '</div>\n'
    )


def create_html_validation(model, classes, data, output_path, matrix_and_wrong_pred=None, grad_cam=False, gif_ms=100,
                           acc_loss=None, verbose=True):
    if verbose:
        print('[I] Creating the HTML validation file.')

    # Get input shape of model
    input_shape = model.get_input_shape_at(0)

    # Get number of input frames needed for single decision
    if len(input_shape) == 5:
        # Model type is LSTM
        num_frames = input_shape[1]
        if grad_cam is True:
            print(W_CREATE_HNTML_GC_NOT_SUP)
            grad_cam = False
    else:
        # Model is Time-Independent
        num_frames = 1
    # endif  len(input_shape) == 5 // Get number of input frames needed for single decision

    # LSTM Network - create directory for the html files with the images that were used to predict
    if num_frames > 1:
        if not os.path.exists(os.path.join(output_path, 'predicts')):
            os.mkdir(os.path.join(output_path, 'predicts'))
        if not os.path.exists(os.path.join(output_path, 'gifs')):
            os.mkdir(os.path.join(output_path, 'gifs'))

    # The images will be copied to the HTML file
    if not os.path.exists(os.path.join(output_path, 'imgs')):
        os.mkdir(os.path.join(output_path, 'imgs'))

    # The output of Grad CAM
    if grad_cam and not os.path.exists(os.path.join(output_path, 'focuses')):
        os.mkdir(os.path.join(output_path, 'focuses'))

    # Copy template and rename it
    shutil.copy(os.path.join(os.path.dirname(__file__), '..', EVALTEMP), os.path.join(output_path, 'evaluation.html'))

    # copy acc and loss graphs
    if acc_loss is not None:
        shutil.copy(acc_loss[0], os.path.join(output_path, 'acc.svg'))
        shutil.copy(acc_loss[1], os.path.join(output_path, 'loss.svg'))

        acc_loss = ['./acc.svg', './loss.svg']  # store new paths

    # Compute the confusion matrix and save prediction of wrong classification
    if matrix_and_wrong_pred is None:
        matrix, wrong_pred = get_confusion_matrix(model, data)
    else:
        matrix = matrix_and_wrong_pred[0]
        wrong_pred = matrix_and_wrong_pred[1]

    # Write the validation evaluation to the html file
    with open(os.path.join(output_path, 'evaluation.html'), 'r+') as html_file:
        # Load the template and save it to the html file
        template('evaluation.prepare_to_predict', html_file, classes=classes, validation=True, matrix=matrix,
                 acc_loss=acc_loss)

        # Variables for LSTM
        predict_file = ''       # The name of the html file with all frames that were used to predict

        num_total_imgs = len(wrong_pred['position'])  # The total number of images with wrong prediction

        # Variable for Grad CAM
        heat_file = ''          # The name of the img file that shows what the Neural Network sees
        img = None
        img_name = ''

        # Read indexes of wrong predictions
        t_last = time.time()
        for num_processed, pos in enumerate(wrong_pred['position']):
            input_imgs = {
                'path': list(),
                'len': 0
            }

            if verbose and (time.time() - t_last) > PRINT_STATUS:
                # how many images was done
                print(f'[%3d %%] done' % (num_processed * 100 / num_total_imgs))
                t_last = time.time()

            if num_frames > 1:
                img_dir = f'%06.d' % num_processed
                if not os.path.exists(os.path.join(output_path, 'imgs', img_dir)):
                    os.mkdir(os.path.join(output_path, 'imgs', img_dir))

                for i in range(0, num_frames):
                    img = data[0][pos, i, ...]
                    img_name = f'%03.d.jpg' % i
                    path_to_img = os.path.join('imgs', img_dir, img_name)
                    cv2.imwrite(os.path.join(output_path, path_to_img), np.multiply(img, 255))

                    input_imgs['path'].append(path_to_img)
                    input_imgs['len'] += 1
            else:
                img = data[0][pos, ...]
                img_name = f'%06.d.jpg' % num_processed
                path_to_img = os.path.join('imgs', img_name)
                cv2.imwrite(os.path.join(output_path, path_to_img), np.multiply(img, 255))

                input_imgs['path'].append(path_to_img)
                input_imgs['len'] += 1

            prediction = wrong_pred['prediction'][num_processed]
            decision = np.argmax(prediction)

            gifname = None
            if num_frames > 1:
                predict_file = f'%06.d' % num_processed

                # Create gif
                gif = []
                for i in input_imgs['path']:
                    gif.append(Image.open(os.path.join(output_path, i)))

                gifname = os.path.join('gifs', predict_file + '.gif')
                gif[0].save(os.path.join(output_path, gifname), save_all=True, append_images=gif[1:], duration=gif_ms,
                            loop=0)
                gifname = './' + gifname

                with open(os.path.join(output_path, 'predicts', predict_file + '.html'), 'w') as p_file:
                    template('predict.img_list', p_file, path=input_imgs['path'], back_step='./../')
            elif grad_cam:
                # What the Neural Network see - it is not supported for LSTM network
                heat_map = visualize_cam(model, -1, decision, img)
                heat_file = os.path.join('focuses', img_name)
                cv2.imwrite(os.path.join(output_path, heat_file), heat_map)
                heat_file = './' + heat_file
            # Saving images to the directory

            template('evaluation.predict', html_file, predict_file=predict_file, output_vector=prediction,
                     true_label=data[1][pos], gif=gifname, classes=classes, view=('./' + input_imgs['path'][0]),
                     grad_cam=grad_cam, heat_file=heat_file)

    if verbose:
        print('[I] The HTML validation file was successfully created.')
