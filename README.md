
# Test task for Internship in Data Science

### Airbus Ship Detection Challenge

This notebook contains my solution Kaggle's [Airbus Ship Detection challenge](https://www.kaggle.com/c/airbus-ship-detection/code?competitionId=9988&sortBy=scoreDescending&language=Python). 


This task requires you to find ships in the images and place an aligned bounding box segment around the ships. 
Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc. 

The solution includes several files:

	Data_exploration_and_analysis.ipynb

Exploration and analysis of images. 

	Unet.py

Native keras implementation of Unet architecture.

	configs.py

Config file to easy change parameters.

	inference.py
Run prediction of trained model on first image from test_2 directory.

	my_utils.py
Function library for decoing masks, generating images for training, etc.

	requirements.tx
Contains necessary package requirements for solution.

	training.py
Run data preparation, augmentation and traing of model with training history graphs as output.
