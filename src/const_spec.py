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

# Parameters for optimization
EPOCHS = 20  # number of epochs for experiments

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
# - ArchitectureName: VGG16, InceptionResNetV2, ...
# - TopLayers: A (Global Average Pooling), M (Global Max Pooling), F (Flatten), D (Dense layer), S (Softmax layer)
# - OptimizerType: SGD, Adam, RMSProp
class Architecture:
    # VGG16 architecture
    class VGG16:
        class FS:
            Adam = 'VGG16_FS_Adam'
            AMSGrad = 'VGG16_FS_AMSGrad'
            SGD = 'VGG16_FS_SGD'
            RMSprop = 'VGG16_FS_RMSprop'

            all = [Adam, AMSGrad, SGD, RMSprop]

        class FDS:
            SGD = 'VGG16_FDS_SGD'
            all = [SGD]

        all = FS.all + FDS.all

    # InceptionResNetV2 architecture
    class InceptionResNetV2:
        class FS:
            SGD = 'InceptionResNetV2_FS_SGD'

            class LSTM:
                Adam = 'InceptionResNetV2_FS_LSTM_Adam'
                AMSGrad = 'InceptionResNetV2_FS_LSTM_AMSGrad'
                SGD = 'InceptionResNetV2_FS_LSTM_SGD'
                RMSprop = 'InceptionResNetV2_FS_LSTM_RMSprop'
                all = [Adam, AMSGrad, SGD, RMSprop]

            all = [SGD] + LSTM.all

        class FDS:
            SGD = 'InceptionResNetV2_FDS_SGD'

            all = [SGD]

        class AS:
            SGD = 'InceptionResNetV2_AS_SGD'

            class LSTM:
                AMSGrad = 'InceptionResNetV2_AS_LSTM_AMSGrad'
                all = [AMSGrad]

            all = [SGD] + LSTM.all

        class ADS:
            SGD = 'InceptionResNetV2_ADS_SGD'
            all = [SGD]

        class MS:
            SGD = 'InceptionResNetV2_MS_SGD'

            class LSTM:
                AMSGrad = 'InceptionResNetV2_MS_LSTM_AMSGrad'
                all = [AMSGrad]

            all = [SGD] + LSTM.all

        class MDS:
            SGD = 'InceptionResNetV2_MDS_SGD'
            all = [SGD]

        all = FS.all + FDS.all + AS.all + ADS.all + MS.all + MDS.all

    # InceptionV3 architecture
    class InceptionV3:
        class FS:
            SGD = 'InceptionV3_FS_SGD'

            all = [SGD]

        class FDS:
            SGD = 'InceptionV3_FDS_SGD'
            all = [SGD]

        all = FS.all + FDS.all

    # MobileNetV2 architecture
    class MobileNetV2:
        class FS:
            SGD = 'MobileNetV2_FS_SGD'
            all = [SGD]

        class FDS:
            SGD = 'MobileNetV2_FDS_SGD'
            all = [SGD]

        class AS:
            SGD = 'MobileNetV2_AS_SGD'

            class LSTM:
                Adam = 'MobileNetV2_AS_LSTM_Adam'
                AMSGrad = 'MobileNetV2_AS_LSTM_AMSGrad'
                SGD = 'MobileNetV2_AS_LSTM_SGD'
                RMSprop = 'MobileNetV2_AS_LSTM_RMSprop'
                all = [Adam, SGD, RMSprop]

            class LSTM16:
                Adam = 'MobileNetV2_AS_LSTM16_Adam'
                all = [Adam]

            class LSTM64:
                Adam = 'MobileNetV2_AS_LSTM64_Adam'
                all = [Adam]

            class LSTM128:
                Adam = 'MobileNetV2_AS_LSTM128_Adam'
                all = [Adam]

            all = [SGD] + LSTM.all + LSTM16.all + LSTM64.all + LSTM128.all

        class ADS:
            SGD = 'MobileNetV2_ADS_SGD'

            class LSTM:
                Adam = 'MobileNetV2_ADS_LSTM_Adam'
                all = [Adam]

            all = [SGD] + LSTM.all

        class MS:
            SGD = 'MobileNetV2_MS_SGD'
            all = [SGD]

        class MDS:
            SGD = 'MobileNetV2_MDS_SGD'
            all = [SGD]

        all = FS.all + FDS.all + AS.all + ADS.all + MS.all + MDS.all


# Batch sizes for each architecture (that are used to training)
BatchSize = {
    # VGG16
    Architecture.VGG16.FS.Adam: 50,
    Architecture.VGG16.FS.AMSGrad: 50,
    Architecture.VGG16.FS.RMSprop: 75,
    Architecture.VGG16.FS.SGD: 75,

    Architecture.VGG16.FDS.SGD: 50,

    # InceptionResNetV2
    Architecture.InceptionResNetV2.FS.SGD: 20,
    Architecture.InceptionResNetV2.FDS.SGD: 20,
    Architecture.InceptionResNetV2.AS.SGD: 20,
    Architecture.InceptionResNetV2.ADS.SGD: 20,
    Architecture.InceptionResNetV2.MS.SGD: 20,
    Architecture.InceptionResNetV2.MDS.SGD: 20,

    Architecture.InceptionResNetV2.FS.LSTM.Adam: 15,
    Architecture.InceptionResNetV2.FS.LSTM.AMSGrad: 15,
    Architecture.InceptionResNetV2.FS.LSTM.SGD: 15,
    Architecture.InceptionResNetV2.FS.LSTM.RMSprop: 15,
    Architecture.InceptionResNetV2.AS.LSTM.AMSGrad: 15,
    Architecture.InceptionResNetV2.MS.LSTM.AMSGrad: 15,

    # InceptionV3
    Architecture.InceptionV3.FS.SGD: 75,
    Architecture.InceptionV3.FDS.SGD: 20,

    # MobileNetV2
    Architecture.MobileNetV2.FS.SGD: 20,
    Architecture.MobileNetV2.FDS.SGD: 10,
    Architecture.MobileNetV2.AS.SGD: 20,
    Architecture.MobileNetV2.ADS.SGD: 15,
    Architecture.MobileNetV2.MS.SGD: 20,
    Architecture.MobileNetV2.MDS.SGD: 15,

    Architecture.MobileNetV2.AS.LSTM.SGD: 15,
    Architecture.MobileNetV2.AS.LSTM.Adam: 15,
    Architecture.MobileNetV2.AS.LSTM.RMSprop: 15,
    Architecture.MobileNetV2.AS.LSTM16.Adam: 15,
    Architecture.MobileNetV2.AS.LSTM64.Adam: 15,
    Architecture.MobileNetV2.AS.LSTM128.Adam: 15,

    Architecture.MobileNetV2.ADS.LSTM.Adam: 15,
}
