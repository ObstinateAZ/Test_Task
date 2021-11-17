from Unet import seg_model
from configs import *
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from skimage.io import imread


# Load saved weights
seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')

# Get list of images from test directory
test_paths = os.listdir(test_image_dir)

# Test first image from list
c_img_name = test_paths[0]

# Read image and prepare it
c_path = os.path.join(test_image_dir, c_img_name)
c_img = imread(c_path)
img = np.expand_dims(c_img, 0) / 255.0

# Get prediction
prediction = seg_model.predict(img)

# Show image and it`s prediction
fig, ax = plt.subplots(1, 2, figsize = (16, 16))
fig.subplots_adjust(wspace=0, hspace=0)

ax[0].axis('off')
ax[0].imshow(img)
ax[0].set_title('Image')

ax[1].axis('off')
ax[1].imshow(prediction[0, :, :, 0], vmin = 0, vmax = 1)
ax[1].set_title('Prediction')