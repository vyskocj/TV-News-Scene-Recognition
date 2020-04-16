import os
import h5py
import time
import keras
import argparse

from src.const_spec import *
import src.evalgen as eg
import src.recognizer as rc
import src.input_converter as ic


pause = 10  # pause between checking and processing arguments in [s]

if __name__ == '__main__':
    """
    Examples: 
    1. fit model VGG16_FS_SGD with verbose and GradCAM for evaluation
    python run.py -m VGG16_FS_SGD -v -F -gc ./dataset.h5 ./output
    
    2. load the model and get evaluation on test data without gradCAM
    python run.py -m "my_label_for_latex" -l ./my_model.h5 -T ./dataset.h5 ./output
    
    3. load the model and get predictions with timeline and overwrite existing data
    python run.py -l ./my_model.h5 -P ./TV_News_images ./output
    
    4. create a new dataset without test data
    python run.py ./train_data ./validation_data None ./output/dataset.h5
    """

    # Parse commandline
    parser = argparse.ArgumentParser(description='Train or evaluate the Neural Network.')

    # Optional arguments
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite files in the output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display extended information')

    parser.add_argument('-m', '--model_type', help='Create new model (see const_spec file - class Architecture).')
    parser.add_argument('-l', '--load_model', help='Load the model which will be used for evaluation.')
    parser.add_argument('-w', '--load_weights', help='Load weights of the model to train LSTM network. This parameter '
                                                     'is not allowed when -l argument is passed!')

    parser.add_argument('-F', '--fit', action='store_true', help='Train a model (see X/Y_TRAIN and X/Y_VALID constants '
                                                                 'in const_spec file for required keywords in the '
                                                                 'dataset file). Argument -gc can be used for a time-'
                                                                 'independent network. A confusion matrix and an HTML '
                                                                 'file with predictions on validation data is '
                                                                 'automatically generated also')
    parser.add_argument('-V', '--valid', action='store_true', help='Generate a Confusion matrix and an HTML file with '
                                                                   'predictions (see X/Y_VALID constant in const_spec '
                                                                   'file). Argument -gc can be used for a time-'
                                                                   'independent network')
    parser.add_argument('-T', '--test', action='store_true', help='Generate a confusion matrix and an HTML file with '
                                                                  'predictions (see X/Y_TEST constant in const_spec '
                                                                  'file). Argument -gc can be used for a time-'
                                                                  'independent network')
    parser.add_argument('-P', '--predict', action='store_true', help='Generate an HTML file with the predictions. '
                                                                     'Argument -gc can be used for a time-independent '
                                                                     'network')
    parser.add_argument('-gc', '--gradCAM', action='store_true', help='Use the GradCAM for the evaluation '
                                                                      '(significantly slows down the program) - this '
                                                                      'feature is supported only for time independent '
                                                                      'networks!')
    # Positional arguments
    parser.add_argument('dataset', nargs='+', help='Path to the dataset. The dataset must be a h5 file for these '
                                                   'arguments: -F, -V, -T; The path must lead to a directory '
                                                   'containing images if argument -P is passed. If there is a request '
                                                   'to create a new dataset, three paths are expected in following '
                                                   'order: <train data> <validation data> <test data>; None shall be '
                                                   'passed if any set should be skipped.')
    parser.add_argument('output', help='Output file for creating dataset, output directory otherwise.')

    # Parsing arguments
    args = parser.parse_args()

    # Summarize
    target = 'https://dspace5.zcu.cz/handle/11025/110/browse?type=author&order=ASC&rpp=20' \
             '&value=Vysko%C4%8Dil%2C+Ji%C5%99%C3%AD'
    print('This script is a part of the thesis: Scene type recognition of TV News broadcasts using visual data.')
    print(f'See Digital Library University of West Bohemia: {target}\n')

    print('Summarize:')
    print(f'Output file/dir:\t%s' % os.path.abspath(args.output))
    print(f'Rewriting file(s):\t%s' % args.overwrite)
    print(f'Verbose:\t\t%s' % args.verbose)

    if len(args.dataset) == 1:
        if args.load_model is not None:
            print(f'Loaded model:\t\t%s' % os.path.abspath(args.load_model))
        elif args.model_type is not None:
            print(f'Used model:\t\t%s' % args.model_type)

        if args.load_weights is not None:
            print(f'Loaded weights:\t\t%s' % os.path.abspath(args.load_weights))

        print(f'Used dataset:\t\t%s' % os.path.abspath(args.dataset[0]))
        print('The following actions will be performed: ')
        if args.fit is True:
            print(f' - fit: containing validation file with' + ('' if args.gradCAM is True else 'out') + ' GradCAM')
        elif args.valid is True:
            print(f' - valid: HTML file with' + ('' if args.gradCAM is True else 'out') + ' GradCAM')

        if args.test is True:
            print(f' - test: HTML file with' + ('' if args.gradCAM is True else 'out') + ' GradCAM')

        if args.predict is True:
            print(f' - predict: HTML file with' + ('' if args.gradCAM is True else 'out') + ' GradCAM')
    else:
        for i in range(0, 3):
            args.dataset[i] = os.path.abspath(args.dataset[i]) if args.dataset[i] != 'None' else None

        print(f'Dataset will be created:\n - train data:\t\t%s\n - validation data:\t%s\n - test data:\t\t%s' %
              (args.dataset[0], args.dataset[1], args.dataset[2]))

    # Arguments checking
    evaluation_dir = {
        'fit': '',
        'test': 'evaluation_test',
        'valid': 'evaluation_valid',
        'predict': 'evaluation_predict'
    }

    if len(args.dataset) == 1:
        # output argument
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        elif not os.path.isdir(args.output):
            raise Exception('[E] The output argument cannot contain a file name!')

        # fit, valid, test and predict argument
        if args.fit is True or args.valid is True or args.test is True:
            if args.predict is True:
                raise Exception('[E] Arguments -F, -V and -T are expecting a h5 dataset; argument -P is expecting a '
                                'directory containing images!')
            elif os.path.isdir(args.dataset[0]):
                raise Exception('[E] The dataset must be a h5 file!')
        elif args.predict is True and not os.path.isdir(args.dataset[0]):
            raise Exception('[E] The dataset must be a directory containing images!')

        if args.fit is True and args.model_type is not None:
            path = os.path.join(args.output, args.model_type)
            if not os.path.exists(path):
                os.mkdir(path)
            # when recognizer.train_model() is called, new directory with the date is made => nothing can be overwritten
            evaluation_dir['fit'] = args.model_type

        if args.predict is True:
            path = os.path.join(args.output, evaluation_dir['predict'])
            if not os.path.exists(path):
                os.mkdir(path)
            elif os.listdir(path) != [] and args.overwrite is False:
                raise Exception(f'[E] "{path}" is not empty!')

        # load_model and model_type argument
        if args.load_model is not None:
            if not os.path.exists(args.load_model):
                raise Exception('[E] Path to the model does not exist!')
            elif args.fit is True:
                raise Exception('[E] This script does not support training of the loaded model!')
        elif args.load_model is None and args.model_type is None:
            raise Exception('[E] Model not selected!')
        elif args.load_weights is not None and args.model_type is None:
            raise Exception('[E] You must select LSTM model (see -m argument) if you want to load weights!!')
    else:
        # Creating dataset
        # output argument
        if os.path.isdir(args.output):
            raise Exception('[E] The output argument must contain a file name!')

        # overwrite argument
        if args.overwrite is False and os.path.exists(args.output):
            raise Exception('[E] Rewriting output file is not allowed! Pass argument -ow to overwrite dataset!')

        # dataset argument
        if args.dataset == [None, None, None]:
            raise Exception('[E] Dataset cannot be created, because all paths are None!')

    # Pause for checking the parameters
    print(f'\nThe program will continue in {pause} seconds...')
    time.sleep(pause)

    # Arguments processing
    if len(args.dataset) == 1:
        # load_model or model_type argument
        if args.load_model is not None:
            model = keras.models.load_model(args.load_model)
            if args.model_type is not None:
                label = args.model_type
            else:
                label = model.name

            optimizer, batch_size, epochs = None, None, None
        else:
            model, optimizer, batch_size, epochs = rc.create_model(args.model_type, args.load_weights)
            label = args.model_type

        # fit, valid and test arguments
        if args.fit is True or args.valid is True or args.test is True:
            with h5py.File(args.dataset[0], 'r') as dataset:
                # fit argument
                if args.fit is True and optimizer is not None:
                    x_train = dataset[X_TRAIN]
                    y_train = keras.utils.to_categorical(dataset[Y_TRAIN], NUM_CLASSES)

                    x_valid = dataset[X_VALID]
                    y_valid = keras.utils.to_categorical(dataset[Y_VALID], NUM_CLASSES)

                    rc.train_model(model, (x_train, y_train), (x_valid, y_valid), epochs, batch_size, optimizer,
                                   output_path=os.path.join(args.output, evaluation_dir['fit']), tex_label=label,
                                   verbose=args.verbose)

                # valid argument - (evaluation file is created by fitting the model for given validation dataset)
                elif args.valid is True:
                    # set the output path
                    path = os.path.join(args.output, model.name)
                    if not os.path.exists(path):
                        os.mkdir(path)

                    path = os.path.join(path, evaluation_dir['valid'])
                    if not os.path.exists(path):
                        os.mkdir(path)
                    elif os.listdir(path) != [] and args.overwrite is False:
                        raise Exception(f'[E] "{path}" is not empty!')

                    # create the evaluation file
                    x_valid = dataset[X_VALID]
                    y_valid = keras.utils.to_categorical(dataset[Y_VALID], NUM_CLASSES)

                    matrix, wrong_pred_vector = eg.get_confusion_matrix(model, (x_valid, y_valid))

                    eg.create_tex_validation(matrix, CLASS_NAMES, os.path.join(args.output, evaluation_dir['valid']),
                                             label=label, verbose=args.verbose)
                    eg.create_html_validation(model, CLASS_NAMES, (x_valid, y_valid),
                                              os.path.join(args.output, evaluation_dir['valid']), grad_cam=args.gradCAM,
                                              matrix_and_wrong_pred=(matrix, wrong_pred_vector), verbose=args.verbose)

                # test argument
                if args.test is True:
                    # set the output path
                    path = os.path.join(args.output, model.name)
                    if not os.path.exists(path):
                        os.mkdir(path)

                    path = os.path.join(path, evaluation_dir['test'])
                    if not os.path.exists(path):
                        os.mkdir(path)
                    elif os.listdir(path) != [] and args.overwrite is False:
                        raise Exception(f'[E] "{path}" is not empty!')

                    # create the evaluation file
                    x_test = dataset[X_TEST]
                    y_test = keras.utils.to_categorical(dataset[Y_TEST], NUM_CLASSES)

                    matrix, wrong_pred_vector = eg.get_confusion_matrix(model, (x_test, y_test))

                    eg.create_tex_validation(matrix, CLASS_NAMES, os.path.join(args.output, evaluation_dir['test']),
                                             label=label, verbose=args.verbose)
                    eg.create_html_validation(model, CLASS_NAMES, (x_test, y_test),
                                              os.path.join(args.output, evaluation_dir['test']), grad_cam=args.gradCAM,
                                              matrix_and_wrong_pred=(matrix, wrong_pred_vector), verbose=args.verbose)

        # predict argument
        elif args.predict is True:
            eg.create_html(model, CLASS_NAMES, args.dataset[0], os.path.join(args.output, evaluation_dir['predict']),
                           portable=True, grad_cam=args.gradCAM, verbose=args.verbose)
    else:
        # Creating dataset
        # check img_shape of the dataset
        existing_path = args.dataset[0] if args.dataset[0] is not None else (
            args.dataset[1] if args.dataset[1] is not None else args.dataset[2]
        )
        existing_path = os.path.join(existing_path, os.listdir(existing_path)[0])

        if os.path.isdir(os.path.join(existing_path, os.listdir(existing_path)[0])):
            img_shape = INPUT_SHAPE_TD
            lstm = True
        else:
            img_shape = INPUT_SHAPE
            lstm = False

        ic.create_dataset(args.output, train=args.dataset[0], valid=args.dataset[1], test=args.dataset[2],
                          img_shape=img_shape, lstm=lstm, verbose=args.verbose)
