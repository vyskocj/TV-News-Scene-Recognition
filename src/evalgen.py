import os
import shutil
import cv2
import json
import re
import numpy as np
import matplotlib.pyplot as plt


E_TEMPLATE_CLASSES = '[E] Pass the parameter to the template function: ' \
                     '"classes" - a list of neural network output names.'
E_TEMPLATE_VIEW = '[E] Pass the parameter to the template function: ' \
                  '"view" - path to the image / gif that represents prediction.'
E_TEMPLATE_OUTPUT_VECTOR = '[E] Pass the parameter to the template function: ' \
                           '"output_vector" - the output of predict method.'
E_TEMPLATE_PATH_OR_GHOST = '[E] Pass the parameter to the template function: ' \
                           '"path" or "ghost" - list of image paths.'
E_REPLACE_TAG_NOT_FOUND = '[E] The tag not found in template.'


def get_time(filename):
    time = re.search(r'(\d+)ms', filename)
    if time is not None:
        time = f'%3d:%02d:%03d [min:s:ms]' % \
               (int(time[1]) / 60_000, (int(time[1]) / 1000) % 60, (int(time[1]) % 1000))
    else:
        time = ''

    return time


def template(part, file, **kwargs):
    if part == 'evaluation.prepare_to_predict':
        if 'classes' not in kwargs.keys():
            raise Exception(E_TEMPLATE_CLASSES)

        replace(string=str(kwargs['classes']), old_tag='classArray', file=file)

        nav_bar = ''
        class_table = ''
        for c in kwargs['classes']:
            nav_bar += '        <a href="javascript:void(0)" id="%s" onclick="showById(\'%s\')">%s</a>\n' % (c, c, c)
            class_table += '            <td style="min-width:100px">' + c + '</td>\n'

        replace(string=nav_bar[8:-1], old_tag='nav_bar', file=file)             # 8  -> number of spaces, -1 -> '\n'
        replace(string=class_table[12:-1], old_tag='class_table', file=file)    # 12 -> number of spaces, -1 -> '\n'
        replace(string='', old_tag='evaluation_predict', file=file)

    elif part == 'evaluation.predict':
        if 'view' not in kwargs.keys():
            raise Exception(E_TEMPLATE_VIEW)

        elif 'output_vector' not in kwargs.keys():
            raise Exception(E_TEMPLATE_OUTPUT_VECTOR)

        elif 'classes' not in kwargs.keys():
            raise Exception(E_TEMPLATE_CLASSES)

        view = ' ' * 16 + '<img src="' + kwargs['view'] + '" width="100%" title="' + get_time(kwargs['view']) + '" />\n'
        if 'predict_file' in kwargs.keys() and kwargs['predict_file'] != '':
            view = '                <a href=".\\predicts\\' + kwargs['predict_file'] + '.html" target="_blank">\n' \
                   '    ' + view + \
                   '                </a>\n'

        output_vector = ''
        for decision in kwargs['output_vector']:
            output_vector += '            <td>' + ('<b>' if decision == max(kwargs['output_vector']) else '') \
                             + (f'%.6f' % decision) + \
                             ('</b>' if decision == max(kwargs['output_vector']) else '') + '</td>\n'

        string = (
            '        <tr class="' + kwargs['classes'][np.argmax(kwargs['output_vector'])] + '">\n'
            '            <td>\n'
            + view +
            '            </td>\n'
            + output_vector +
            '            <td>' + kwargs['classes'][np.argmax(kwargs['output_vector'])] + '</td>\n'
            '        </tr>\n'
        )

        insert(string, file)

    elif part == 'predict.img_list':
        if 'path' not in kwargs.keys() or 'ghost' not in kwargs.keys():
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

    return '..\\' * (len(output_step) - input_step.count('..'))  # multiplying by a negative number does not matter


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


def create_html(model, classes, input_path, output_path, files_together=False):

    # Get input shape of model
    input_shape = model.get_input_shape_at(0)

    # Get number of input frames needed for single decision
    if len(input_shape) == 5:
        # Model type is LSTM
        num_frames = input_shape[1]
    else:
        # Model is Time-Invariant
        num_frames = 1
    # endif  len(input_shape) == 5 // Get number of input frames needed for single decision

    # Get image shape
    img_shape = input_shape[-3:]

    # LSTM Network - create directory for the html files with the images that were used to predict
    if num_frames > 1 and not os.path.exists(output_path + '\\predicts'):
        os.mkdir(output_path + '\\predicts')

    if files_together and not os.path.exists(output_path + '\\imgs'):
        os.mkdir(output_path + '\\imgs')

    # Copy template and rename it
    shutil.copy('..\\temp\\evaltemp.html', output_path + '\\evaluation.html')

    # Write the evaluation to the html file
    with open(output_path + '\\evaluation.html', 'r+') as html_file:
        # Load the template and save it to the html file
        template('evaluation.prepare_to_predict', html_file, classes=classes)

        # Compute how many back steps it takes from html file to the images
        if files_together:
            back_step = '.\\'
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
        num_dirs_in_path = 0    # Counter that rises with each detection that a path leads to a directory
        num_files_in_input_path = len(os.listdir(input_path))  # The total number of files in the input path

        # Read data from the given directory
        for file_name in os.listdir(input_path):
            path = input_path + '\\' + file_name

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
                num_imgs_in_dir = num_files_in_input_path - num_dirs_in_path  # Compute how many imgs can be in the dir
            # endif os.path.isdir(path) // Check if the file is directory

            for img_name in directory:
                # Load image and prepare it for model prediction
                path_to_img = path + '\\' + img_name
                img = cv2.imread(path_to_img)
                img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
                if files_together:
                    path_to_img = 'imgs\\' + img_name
                    cv2.imwrite(output_path + '\\' + path_to_img, img)
                img = img.astype("float32")
                img = np.divide(img, 255.0)

                # Lists length increment, save the image and its path
                input_imgs['path'].append(path_to_img)
                input_imgs['imgs'].append(img)
                input_imgs['len'] += 1

                # Check if images can be passed to the network
                if input_imgs['len'] < num_frames and input_imgs['len'] < (num_imgs_in_dir - pivot * num_frames):
                    # There are still some images that can be loaded
                    # Save last image name - it represents when the image was taken - and continue
                    last_img_name = img_name
                    continue
                elif input_imgs['len'] > num_frames:
                    # More images are stored in the memory than can be passed through the network
                    # Delete the "oldest" image
                    # TODO: Zatím není použito, input_imgs se mažou níže u TODO
                    input_imgs['path'].pop(0)
                    input_imgs['imgs'].pop(0)
                    input_imgs['len'] -= 1

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
                    # Time-Invariant network
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

                # TODO: přidat gif pokud LSTM, zobrazovat jen obrázek pokud není LSTM
                if num_frames > 1:
                    predict_file = f'%06.d' % file_index
                    with open(output_path + '\\predicts\\' + predict_file + '.html', 'w') as p_file:
                        template('predict.img_list', p_file, path=input_imgs['path'], ghost=input_imgs['ghost'],
                                 back_step=(back_step + "..\\"))
                    file_index += 1

                template('evaluation.predict', html_file, predict_file=predict_file, output_vector=network_say,
                         classes=classes, view=(back_step + input_imgs['path'][0]))

                # TODO: předělat na strides - parametr funkce
                input_imgs = {
                    'path': list(),
                    'imgs': list(),
                    'ghost': list(),
                    'len': 0
                }
                pivot += 1

                # Change the last image name if model is Time-Invariant
                if num_frames == 1:
                    last_img_name = img_name

        with open(output_path + '\\decisions.json', 'w') as json_file:
            decisions['class'] = np.array(decisions['class'], dtype='int8')
            decisions['time'] = np.array(decisions['time'], dtype='int64')

            decisions['class'] = decisions['class'].astype('int8').tolist()
            decisions['time'] = decisions['time'].astype('int64').tolist()

            json.dump(decisions, json_file)

        plt.figure(figsize=(20, 5))

        time = [x / 60_000 for x in decisions['time']]

        plt.plot(time, decisions['class'])
        plt.xticks(np.arange(min(time), max(time), 5))
        plt.yticks(np.arange(len(classes)), classes)

        # Show the major grid lines with dark grey lines
        plt.grid(b=True, which='major', color='#7F7F7F', linestyle='-')

        # Show the minor grid lines with bright grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#C0C0C0', linestyle='-', alpha=0.3)

        plt.xlim((min(time), max(time)))

        plt.title('Timeline of decisions')
        plt.ylabel('Classification')
        plt.xlabel('Time [min]')
        plt.legend(['Decision'], loc='upper left')

        plt.savefig(output_path + '\\timeline.svg')
