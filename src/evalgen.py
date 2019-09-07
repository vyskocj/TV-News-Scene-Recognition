import os
import cv2
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from src.const_spec import *

E_TEMPLATE_CLASSES = '[E] Pass the parameter to the template function: ' \
                     '"classes" - a list of neural network output names.'
E_TEMPLATE_VIEW = '[E] Pass the parameter to the template function: ' \
                  '"view" - path to the image / gif that represents prediction.'
E_TEMPLATE_OUTPUT_VECTOR = '[E] Pass the parameter to the template function: ' \
                           '"output_vector" - the output of predict method.'
E_TEMPLATE_PATH_OR_GHOST = '[E] Pass the parameter to the template function: ' \
                           '"path" or "ghost" - list of image paths.'


def get_time(filename):
    time = re.search(r'(\d+)ms', filename)
    if time is not None:
        time = f'%3d:%02d:%03d [min:s:ms]' % \
               (int(time[1]) / 60_000, (int(time[1]) / 1000) % 60, (int(time[1]) % 1000))
    else:
        time = ''

    return time


def template(part, **kwargs):
    if part == 'evaluation_top':
        if 'classes' not in kwargs.keys():
            print(E_TEMPLATE_CLASSES)
            return MISSING_PARAMETER

        classes = ''
        nav_bar = ''
        for c in kwargs['classes']:
            classes += '        <td style="min-width:100px">' + c + '</td>\n'
            nav_bar += '      <a href="javascript:void(0)" id="%s" onclick="showById(\'%s\')">%s</a>\n' % (c, c, c)

        return (
            # TODO: Navigační menu by mohlo být zafixováno při scrollování
            # TODO: Po kliknutí na timeline obrázek se otevře v novém okně
                '<html>\n'
                '  <head>\n'
                '    <!-- The navigation bar was inspired by: https://www.w3schools.com/howto/howto_js_sidenav.asp -->\n\n'
                '    <title>Evaluation</title>\n'
                '    <style>\n'

                '      #navigation_bar {\n'
                '        height: 100%;\n'
                '        width: 0;\n'
                '        position: fixed;\n'
                '        top: 0;\n'
                '        left: 0;\n'
                '        background-color: black;\n'
                '        overflow-x: hidden;\n'
                '        transition: 0.05s;\n'
                '        padding-top: 75px;\n'
                '      }\n\n'

                '      #navigation_bar a {\n'
                '        padding: 5px 30px 5px 30px;\n'
                '        text-decoration: none;\n'
                '        font-size: 20px;\n'
                '        color: lightgrey;\n'
                '        display: block;\n'
                '        transition: 0.1s;\n'
                '      }\n\n'

                '      #navigation_bar a:hover {\n'
                '        color: gold;\n'
                '      }\n\n'

                '      #navigation_bar #close_navigation_bar {\n'
                '        position: absolute;\n'
                '        top: 0;\n'
                '        right: 0px;\n'
                '        font-size: 40px;\n'
                '      }\n\n'

                '      #navigation_bar span {\n'
                '        position: absolute;\n'
                '        margin-top: 7px;\n'
                '        padding: 5px 30px 5px 20px;\n'
                '        text-decoration: none;\n'
                '        display: block;\n'
                '        top: 0;\n'
                '        left: 0px;\n'
                '        font-size: 40px;\n'
                '        color: lightgrey;\n'
                '      }\n\n'

                '    </style>\n'
                '    <script>\n'
                '      var navBarChecked = \'All\';\n'
                '      var classArray = ' + str(kwargs['classes']) + ';\n\n'

                                                                     '      function openNavBar() {\n'
                                                                     '        document.getElementById("navigation_bar").style.width = "300px";\n'
                                                                     '      }\n\n'

                                                                     '      function closeNavBar() {\n'
                                                                     '        document.getElementById("navigation_bar").style.width = "0";\n'
                                                                     '      }\n\n'

                                                                     '      function showById(ID) {\n'
                                                                     '        document.getElementById(navBarChecked).removeAttribute("style")\n'
                                                                     '        document.getElementById(ID).style.color = "red";\n\n'

                                                                     '        navBarChecked = ID;\n'
                                                                     '        closeNavBar();\n\n'

                                                                     '        for (var c = 0; c < classArray.length; c++) {\n'
                                                                     '          for (var i = 0; i < document.getElementsByClassName(classArray[c]).length; i++) {\n'
                                                                     '            if ((classArray[c] == ID) || (ID == \'All\'))\n'
                                                                     '              document.getElementsByClassName(classArray[c])[i].removeAttribute("style");\n'
                                                                     '            else\n'
                                                                     '              document.getElementsByClassName(classArray[c])[i].style.display = "none";\n'
                                                                     '          }\n'
                                                                     '        }\n'
                                                                     '      }\n'

                                                                     '    </script>\n'
                                                                     '  </head>\n'

                                                                     '  <body>\n'
                                                                     '    <span style="font-size:40px; cursor:pointer" onclick="openNavBar()">&#9776;</span>\n'
                                                                     '    <div id="navigation_bar">\n'
                                                                     '      <span>Show:</span>\n'
                                                                     '      <a href="javascript:void(0)" id="close_navigation_bar" onclick="closeNavBar()">&#10094;</a>\n'
                                                                     '      <a href="javascript:void(0)" id="All" onclick="showById(\'All\')" style="color: red;">All</a>\n'
                + nav_bar +
                '    </div>\n    <br />\n'
                '    <img src=".\\timeline.svg" width="100%" />\n'
                '    <table border="2">\n'
                '      <tr>\n'
                '        <td>Image</td>\n'
                + classes +
                '        <td>Classification</td>\n'
                '      </tr>\n'
        )

    elif part == 'evaluation_predict':
        if 'view' not in kwargs.keys():
            print(E_TEMPLATE_VIEW)
            return MISSING_PARAMETER

        elif 'output_vector' not in kwargs.keys():
            print(E_TEMPLATE_OUTPUT_VECTOR)
            return MISSING_PARAMETER

        elif 'classes' not in kwargs.keys():
            print(E_TEMPLATE_CLASSES)
            return MISSING_PARAMETER

        view = '          <img src="' + kwargs['view'] + '" width="100%" title="' + get_time(kwargs['view']) + '" />'
        if 'predict_file' in kwargs.keys() and kwargs['predict_file'] != '':
            view = '          <a href=".\\predicts\\' + kwargs['predict_file'] + '.html" target="_blank">' \
                                                                                 '  ' + view + \
                   '          </a>'

        output_vector = ''
        for decision in kwargs['output_vector']:
            output_vector += '        <td>' + ('<b>' if decision == max(kwargs['output_vector']) else '') \
                             + (f'%.6f' % decision) + \
                             ('</b>' if decision == max(kwargs['output_vector']) else '') + '</td>\n'

        return (
                '      <tr class="' + kwargs['classes'][np.argmax(kwargs['output_vector'])] + '">\n'
                                                                                              '        <td>\n'
                + view +
                '        </td>\n'
                + output_vector +
                '        <td>' + kwargs['classes'][np.argmax(kwargs['output_vector'])] + '</td>\n'
                                                                                         '      </tr>\n'
        )

    elif part == 'evaluation_bot':
        return (
            '    </table>\n'
            '  </body>\n'
            '</html>'
        )

    elif part == 'predict_img_list':
        if 'path' not in kwargs.keys() or 'ghost' not in kwargs.keys():
            print(E_TEMPLATE_PATH_OR_GHOST)
            return MISSING_PARAMETER

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

        return (
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


def create_html(model, classes, input_path, output_path):
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

    # Write the evaluation to the html file
    with open(output_path + '\\evaluation.html', 'w') as html_file:
        # Load the template and save it to the html file
        html_file.write(template('evaluation_top', classes=classes))

        # Compute how many back steps it takes from html file to the images
        back_step = get_back_step(input_path, output_path)

        # Lists of input images that going to be passed to the Neural Network
        input_imgs = {
            'path': list(),  # path to the image
            'imgs': list(),  # image that is passed thought network
            'ghost': list(),  # path to the image that was used to complete the required number of input images
            'len': 0
        }

        # The timeline of decisions
        decisions = {
            'class': list(),
            'time': list()
        }
        last_decision = None  # Last classification
        last_img_name = ''  # Last image name -> it is named after the time in milliseconds it was taken

        # Variables for LSTM
        file_index = 0  # Index of the html file with all frames that were used to predict
        predict_file = ''  # The name of the html file with all frames that were used to predict
        pivot = 0  # Pivot is used to identify how many images can be used to predict
        num_dirs_in_path = 0  # Counter that rises with each detection that a path leads to a directory
        num_files_in_input_path = len(os.listdir(input_path))  # The total number of files in the input path

        # Read data from the given directory
        for file_name in os.listdir(input_path):
            path = input_path + '\\' + file_name

            # Check if the file is directory
            if os.path.isdir(path):
                # The directory represents one scene
                pivot = 0  # Pivot is changed only when directory has changed
                directory = os.listdir(path)  # List files in directory
                num_imgs_in_dir = len(os.listdir(path))  # Number of images in the directory
                num_dirs_in_path += 1  # Increment number of directories in path
            else:
                # The file is not a directory but an image
                path = input_path  # Change the path as input_path
                directory = [file_name]  # Pass image name thought for cycle
                num_imgs_in_dir = num_files_in_input_path - num_dirs_in_path  # Compute how many imgs can be in the dir
            # endif os.path.isdir(path) // Check if the file is directory

            for img_name in directory:
                # Load image and prepare it for model prediction
                img = cv2.imread(path + '\\' + img_name)
                img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
                img = img.astype("float32")
                img = np.divide(img, 255.0)

                # Lists length increment, save the image and its path
                input_imgs['path'].append(path + '\\' + img_name)
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
                    for i in range(0, num_frames - input_imgs['len']):
                        input_imgs['imgs'].append(input_imgs['imgs'][-1])
                        input_imgs['ghost'].append(input_imgs['path'][-1])

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
                        p_file.write(template('predict_img_list', path=input_imgs['path'], ghost=input_imgs['ghost'],
                                              back_step=(back_step + "..\\")))
                    file_index += 1

                html_file.write(template('evaluation_predict', predict_file=predict_file, output_vector=network_say,
                                         classes=classes, view=(back_step + input_imgs['path'][0])))

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

        plt.figure(figsize=(15, 5))

        time = [x / 60000 for x in decisions['time']]

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

        html_file.write(template('evaluation_bot'))
