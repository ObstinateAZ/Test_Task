from configs import *
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Unet import seg_model
from my_utils import make_image_gen, dice_p_bce, dice_coef, true_positive_rate, create_aug_gen, show_loss, sample_ships


# Read dataframe from file
df = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))

# Create column for ship count in image
df['ships'] = df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
# Get unique images (no duplicates because of multiple EncodedPixels for one image)
unique_img_ids = df.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
# Create vector to know if image has ships
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
# If sorted by image size, many of small ones are corrupted\have only clouds, so we don`t need them
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                               os.stat(os.path.join(train_image_dir,
                                                                                    c_img_id)).st_size / 1024)
# Keep only 50kb files
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > MIN_FILE_SIZE]
# We no longer need hip count in original dataset
df.drop(['ships'], axis=1, inplace=True)
# Split unique images into training and validation data with same proportion of ship amount
train_ids, valid_ids = train_test_split(unique_img_ids, test_size=0.3, stratify=unique_img_ids['ships'])

train_df = pd.merge(df, train_ids)
valid_df = pd.merge(df, valid_ids)

# Undersample the empty images to get a better balanced group with more ships to try and segment
train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x + 1) // 2).clip(0, 7)
balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)

# Validation Set
valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

# Set augmentation parameters from configs
dg_args = dict(featurewise_center=FEATUREWISE_CENTER,
               samplewise_center=SAMPLEWISE_CENTER,
               rotation_range=ROTATION_RANGE,
               width_shift_range=WIDTH_SHIFT_RANGE,
               height_shift_range=HEIGHT_SHIFT_RANGE,
               shear_range=SHEAR_RANGE,
               zoom_range=ZOOM_RANGE,
               horizontal_flip=HORIZONTAL_FLIP,
               vertical_flip=VERTICAL_FLIP,
               fill_mode=FILL_MODE,
               data_format=DATA_FORMAT,
               brightness_range=BRIGHTNESS_RANGE)

# Brightness can be problematic since it seems to change the labels differently from the images
if not AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)

# Set parameters for model training
seg_model.compile(optimizer=Adam(1e-4, decay=1e-6),
                  loss=dice_p_bce,
                  metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

checkpoint = ModelCheckpoint(weight_path,
                             monitor='val_dice_coef',
                             verbose=1,
                             save_best_only=True,
                             mode='max',
                             save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef',
                                   factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max',
                                   min_delta=0.0001,
                                   cooldown=2,
                                   min_lr=1e-6)

early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=20)

callbacks_list = [checkpoint, early, reduceLROnPlat]

# Set steps per epoch and generate training data
step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0] // BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen(balanced_train_df), image_gen, label_gen)
# Train model
loss_history = [seg_model.fit(aug_gen,
                              steps_per_epoch=step_count,
                              epochs=NB_EPOCHS,
                              validation_data=(valid_x, valid_y),
                              callbacks=callbacks_list,
                              workers=1)]
# print graphs for training history
show_loss(loss_history)
