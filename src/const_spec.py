# Return values of functions
INVALID_OUTPUT_PATH = -1
INVALID_DATA_TYPE = -2
INVALID_INPUT_ARGUMENT = -3

# Shape for default Neural Network
NUM_FRAMES = 25         # the number of frames that LSTM classifies
FRAME_WIDTH = 320       # width of each frame / image
FRAME_HEIGHT = 180      # height of each frame / image
FRAME_CHANNELS = 3      # number of channels, i.e. 3 = RGB, 1 = shades of grey

INPUT_SHAPE = (FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS)   # input shape for model without LSTM
INPUT_SHAPE_TD = (NUM_FRAMES, ) + INPUT_SHAPE               # input shape for time-distributed model, i.e. with LSTM

# Classes for default Neural Network
NUM_CLASSES = 9
