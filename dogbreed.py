"""In a convolutional neural network, the main task of the convolutional
layers is to enhance the important features of an image. If a particular 
filter is used to identify the straight lines in an image then it will
work for other images as well this is particularly what we do in transfer
learning. There are models which are developed by researchers by regress
hyperparameter tuning and training for weeks on millions of images belonging
to 1000 different classes like imagenet dataset. A model that works well for
one computer vision task proves to be good for others as well. Because of this
reason, we leverage those trained convolutional layers parameters and tuned
hyperparameters for our task to obtain higher accuracy."""


"""
Importing Libraries

Python libraries make it very easy for us to handle the data and perform typical and complex tasks with a single line of code.

    Pandas - This library helps to load the data frame in a 2D array format and has multiple functions to perform analysis tasks in one go.
    
    Numpy - Numpy arrays are very fast and can perform large computations in a very short time.
    
    Matplotlib - This library is used to draw visualizations.
    
    Sklearn - This module contains multiple libraries having pre-implemented functions to perform tasks from data preprocessing to model development and evaluation.
    
    OpenCV - This is an open-source library mainly focused on image processing and handling.
    
    Tensorflow - This is an open-source library that is used for Machine Learning and Artificial intelligence and provides a range of functions to achieve complex functionalities with single lines of code.
    
"""

"""
https://www.geeksforgeeks.org/dog-breed-classification-using-transfer-learning/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from functools import partial

import warnings
warnings.filterwarnings('ignore')
AUTO = tf.data.experimental.AUTOTUNE

"""
The dataset which we will use here has been taken from – https://www.kaggle.com/competitions/dog-breed-identification/data. This dataset includes 10,000 images of 120 different breeds of dogs. In this data set, we have a training images folder. test image folder and a CSV file that contains information regarding the image and the breed it belongs to.
"""

from zipfile import ZipFile
data_path = 'dog-breed-identification.zip'

with ZipFile(data_path, 'r') as zip:
	zip.extractall()
	print('The data set has been extracted.')

df = pd.read_csv('labels.csv')
df.head()

df.shape

df['breed'].nunique()


plt.figure(figsize=(10, 5))
df['breed'].value_counts().plot.bar()
plt.axis('off')
plt.show()


plt.subplots(figsize=(10, 10))
for i in range(12):
	plt.subplot(4, 3, i+1)

	# Selecting a random image
	# index from the dataframe.
	k = np.random.randint(0, len(df))
	img = cv2.imread(df.loc[k, 'filepath'])
	plt.imshow(img)
	plt.title(df.loc[k, 'breed'])
	plt.axis('off')
plt.show()


le = LabelEncoder()
df['breed'] = le.fit_transform(df['breed'])
df.head()
