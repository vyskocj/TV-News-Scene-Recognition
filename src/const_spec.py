# Return values of functions
OK = 0
INVALID_OUTPUT_PATH = -1
INVALID_INPUT_PARAMETER = -2
MISSING_PARAMETER = -3

# Time when extended information is printed
PRINT_STATUS = 5  # [sec]

# Dataset keys
X_TRAIN = "train_data"
X_VALID = "valid_data"
X_TEST = "test_data"

Y_TRAIN = "train_labels"
Y_VALID = "valid_labels"
Y_TEST = "test_labels"

# Batch sizes for evalgen file - get_confusion_matrix
BATCH_SIZE = 75    # batch size for time independent models
BATCH_SIZE_TD = 5  # batch size for LSTM model
# Batch sizes for each model are defined bellow

# Shape for default Neural Network
NUM_FRAMES = 25         # the number of frames that LSTM classifies
FRAME_WIDTH = 320       # width of each frame / image
FRAME_HEIGHT = 180      # height of each frame / image
FRAME_CHANNELS = 3      # number of channels, i.e. 3 = RGB, 1 = shades of grey

INPUT_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)   # input shape for model without LSTM
INPUT_SHAPE_TD = (NUM_FRAMES, ) + INPUT_SHAPE               # input shape for time-distributed model, i.e. with LSTM
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)                    # image size in accordance with the OpenCV framework

# Classes for default Neural Network
NUM_CLASSES = 9
CLASS_NAMES = [
    'Graphics',
    'Historic',
    'Indoor',
    'Studio',
    'Mix',
    'Outdoor Country',
    'Outdoor Human Made',
    'Other',
    'Speech'
]

CLASS_NAMES_SH = ['GRA', 'HIS', 'IN', 'STUD', 'MIX', 'OUT_C', 'OUT_H', 'OTHER', 'SPE']

# Path to the template
EVALTEMP = 'temp/evaltemp.html'


# Model architectures: ArchitectureName_TopLayers_OptimizerType_OptimizerParameters
# - ArchitectureName: VGG16, InceptionResNetV2, ..., LSTM_VGG16, ...
# - TopLayers: A (Global Average Pooling), M (Global Max Pooling), F (Flatten), D (Dense layer), S (Softmax layer)
# - OptimizerType: SGD, Adam, RMSProp
class Architecture:
    # VGG16 architecture
    class VGG16:
        class FS:
            SGD = 'VGG16_FS_SGD'
            RMSprop = 'VGG16_FS_RMSprop'
            all = [SGD, RMSprop]

        class FDS:
            SGD = 'VGG16_FDS_SGD'
            all = [SGD]

    # InceptionResNetV2 architecture
    class InceptionResNetV2:
        class FS:
            SGD = 'InceptionResNetV2_FS_SGD'
            all = [SGD]

    # Networks with LSTM layers
    class LSTM:
        class VGG16:
            class FS:
                SGD = 'LSTM_VGG16_FS_SGD'
                all = [SGD]


# Batch sizes for each architecture (that are used to training)
BatchSize = {
    Architecture.VGG16.FS.SGD: 75,
    Architecture.VGG16.FS.RMSprop: 75,
    Architecture.VGG16.FDS.SGD: 75,

    Architecture.InceptionResNetV2.FS.SGD: 20,

    Architecture.LSTM.VGG16.FS.SGD: 5,
}
