import tensorflow as tf
from skimage.io import imread
import matplotlib.pyplot as plt
import keras.backend as K
from keras.losses import binary_crossentropy
import numpy as np  # linear algebra
from configs import *


def rle_decode(mask_rle, shape=IMAGE_SHAPE):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    # Create empty mask vector
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # Decode ship area into mask vector
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # Need to align to RLE direction
    return img.reshape(shape).T


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros(IMAGE_SHAPE, dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            # Add every ship mask to common image mask
            all_masks += rle_decode(mask)
    # Return mask of all ships
    return np.expand_dims(all_masks, -1)


def sample_ships(in_df, base_rep_val=1500):
    if in_df['ships'].values[0] == 0:
        # Strongly undersample no ships images because of severe ship to no ship pixel area unbalance
        return in_df.sample(base_rep_val // 3)
    else:
        # Undersample data with ships (mainly images with only one ship)
        return in_df.sample(base_rep_val, replace=(in_df.shape[0] < base_rep_val))


def make_image_gen(in_df, batch_size=BATCH_SIZE):
    # Make batch image generator based on dataset
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


def create_aug_gen(in_gen, image_gen, label_gen, seed=None):
    # Generate augmented images
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # Keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


def dice_coef(y_true, y_pred, smooth=1):
    # Implementation of Dice score
    y_true = tf.cast(y_true, tf.float32)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


def show_loss(_loss_history):
    # Function to show training history with graphs
    epich = np.cumsum(np.concatenate([np.linspace(0.5, 1, len(mh.epoch)) for mh in _loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in _loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_loss'] for mh in _loss_history]),
                 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in _loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_true_positive_rate'] for mh in _loss_history]),
                 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')

    _ = ax3.plot(epich,
                 np.concatenate([mh.history['binary_accuracy'] for mh in _loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_binary_accuracy'] for mh in _loss_history]),
                 'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')

    _ = ax4.plot(epich,
                 np.concatenate([mh.history['dice_coef'] for mh in _loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_dice_coef'] for mh in _loss_history]),
                 'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')
