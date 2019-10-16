# Return values of functions
OK = 0
INVALID_OUTPUT_PATH = -1
INVALID_INPUT_PARAMETER = -2
MISSING_PARAMETER = -3

# Dataset keys
X_TRAIN = "train_data"
X_TEST = "test_data"
Y_TRAIN = "train_labels"
Y_TEST = "test_labels"

# Batch sizes for training / validation
BATCH_SIZE = 75    # batch size for time independent models
BATCH_SIZE_TD = 5  # batch size for LSTM model

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
