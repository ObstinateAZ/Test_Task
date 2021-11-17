import os

""" Training configs """
BATCH_SIZE = 4
NB_EPOCHS = 5
# number of validation images to use
VALID_IMG_COUNT = 400
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200

""" Image config """
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
IMAGE_SHAPE = (768, 768)
INPUT_SHAPE = (768, 768, 3)
MIN_FILE_SIZE = 50

""" Data augmentation configs """
GAUSSIAN_NOISE = 0.1
EDGE_CROP = 16
AUGMENT_BRIGHTNESS = False
FEATUREWISE_CENTER = False,
SAMPLEWISE_CENTER = False,
ROTATION_RANGE = 15,
WIDTH_SHIFT_RANGE = 0.1,
HEIGHT_SHIFT_RANGE = 0.1,
SHEAR_RANGE = 0.01,
ZOOM_RANGE = [0.9, 1.25],
HORIZONTAL_FLIP = True,
VERTICAL_FLIP = True,
FILL_MODE = 'reflect',
DATA_FORMAT = 'channels_last'
BRIGHTNESS_RANGE = [0.5, 1.5]

""" Paths configs """
ship_dir = 'Data/'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')
weight_path = "{}_weights.best.hdf5".format('seg_model')
