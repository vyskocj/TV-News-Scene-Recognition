import os
import cv2
import numpy as np

from src.const_spec import *


def template(part, classes=None):
    if part == 'evaluation_top':
        if classes is None:
            print('[E] Pass the function parameter: classes = (name_1, name_2,...)')
            return MISSING_PARAMETER

        td_classes = ''
        for i in range(0, len(classes)):
            td_classes += '\t\t\t\t<td style="min-width:100px">' + classes[i] + '</td>\n'

        return (
            '<html>\n'
            '\t<body>\n'
            '\t\t<table border="2">\n'
            '\t\t\t<tr>\n'
            '\t\t\t\t<td>Image</td>\n'
            + td_classes +
            '\t\t\t\t<td>Classification</td>\n'
            '\t\t\t</tr>\n'
        )

    elif part == 'evaluation_bot':
        return (
            '\t\t</table>\n'
            '\t</body>\n'
            '</html>'
        )


def get_back_step(input_path, output_path):
    input_step = input_path.split('\\')
    output_step = output_path.split('\\')

    output_step.remove('..')

    return '..\\' * (len(output_step) - input_step.count('..'))  # multiplying by a negative number does not matter


def create_html(model,  classes, input_path, output_path):
    input_shape = model.get_input_shape_at(0)

    if len(input_shape) == 5:  # lstm
        num_frames = input_shape[1]
    else:
        num_frames = 1

    img_shape = input_shape[-3:]

    with open(output_path + '\\evaluation.html', 'w') as file:
        file.write(template('evaluation_top', classes))
        back_step = get_back_step(input_path, output_path)

        input_imgs = {
            'path': list(),
            'imgs': list()
        }

        index = 0
        for img_name in os.listdir(input_path):
            path = input_path + '\\' + img_name

            img = cv2.imread(path)
            img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
            img = img.astype("float32")
            img = np.divide(img, 255.0)

            input_imgs['path'].append(path)
            input_imgs['imgs'].append(img)
            if len(input_imgs['imgs']) < num_frames:
                continue
            elif len(input_imgs['imgs']) > num_frames:
                input_imgs['path'].pop(0)
                input_imgs['imgs'].pop(0)

            network_say = model.predict(np.array([input_imgs['imgs']]))[0]

            # TODO: připsat do templaty, přidat gif pokud LSTM, zobrazovat jen obrázek pokud není LSTM
            # TODO: maybe - po najetí myší zobrazovat čas snímku
            predict_file = f'%06.d' % index
            with open(output_path + '\\predicts\\' + predict_file + '.html', 'w') as p_file:
                p_file.write('<html>\n\t<body>\n')
                [p_file.write('\t\t<img src="' + back_step + "..\\" + input_imgs['path'][i] + '" width="400px" />\n')
                 for i in range(0, num_frames)]
                p_file.write('\t</body>\n</html>')

            index += 1

            file.write('\t\t\t<tr>\n')
            file.write('\t\t\t\t<td>')
            file.write('<a href=".\\predicts\\' + predict_file + '.html' '">')
            file.write('<img src="' + back_step + input_imgs['path'][0] + '" width="100%" />')
            file.write('</a>')
            file.write('</td>\n')
            [file.write(
                '\t\t\t\t<td>' + ('<b>' if network_say[i] == max(network_say) else '') +
                f"%.6f" % (round(network_say[i] * 1_000_000) / 1_000_000) +
                ('</b>' if network_say[i] == max(network_say) else '') + '</td>\n'
            ) for i in range(0, len(classes))]
            file.write('\t\t\t\t<td>(' + classes[np.argmax(network_say)] + ')')
            file.write('\t\t\t</tr>\n')

            # TODO: předělat na strides - parametr funkce
            input_imgs = {
                'path': list(),
                'imgs': list()
            }

        file.write(template('evaluation_bot'))
