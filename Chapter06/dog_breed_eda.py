
# coding: utf-8

# # Dog Breed Dataset Exploratory Analysis
# 
# This notebook contains basic exploratory analysis of the Stanford Dog Breed Dataset

# In[ ]:

import os
import scipy as sp
import numpy as np
import pandas as pd

import PIL
import scipy.ndimage as spi

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
import seaborn as sns

plt.rcParams.update(params)

np.random.seed(42)


# In[ ]:

# This function prepares a random batch from the dataset
def load_batch(dataset_df, batch_size = 25):
    batch_df = dataset_df.loc[np.random.permutation(np.arange(0,
                                                              len(dataset_df)))[:batch_size],:]
    return batch_df


# In[ ]:

# This function plots sample images in specified size and in defined grid
def plot_batch(images_df, grid_width, grid_height, im_scale_x, im_scale_y):
    f, ax = plt.subplots(grid_width, grid_height)
    f.set_size_inches(12, 12)
    
    img_idx = 0
    for i in range(0, grid_width):
        for j in range(0, grid_height):
            ax[i][j].axis('off')
            ax[i][j].set_title(images_df.iloc[img_idx]['breed'][:10])
            ax[i][j].imshow(sp.misc.imresize(spi.imread(DATASET_PATH + images_df.iloc[img_idx]['id']+'.jpg'),
                                             (im_scale_x,im_scale_y)))
            img_idx += 1
            
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)


# ## Load Dataset Params

# In[ ]:

DATASET_PATH = r'../kaggle_train/'
LABEL_PATH = r'../kaggle_labels/labels.csv'


# In[ ]:

dataset_df = pd.read_csv(LABEL_PATH)
dataset_df.shape


# ## Visualize a Sample Set

# In[ ]:

batch_df = load_batch(dataset_df, 
                    batch_size=36)


# In[ ]:

plot_batch(batch_df, grid_width=6, grid_height=6
           ,im_scale_x=64, im_scale_y=64)


# ## Analyze Image Dimensions

# In[ ]:

file_list = os.listdir(DATASET_PATH)
file_dimension_list = np.asarray([spi.imread(DATASET_PATH + file).shape for file in file_list])
shape_df = pd.DataFrame(file_dimension_list,columns=['width','height','channel'])


# In[ ]:

shape_df.describe()


# Observations:
# * All images are colored (channel is throughout 3)
# * There are some huge images as well _see max for width and height_
# * Most images are taller than being wider

# ### Shape Distribution

# In[ ]:

sns.boxplot(orient="v",data=shape_df[['width','height']])


# ### Scatter Plot

# In[ ]:

plt.plot(file_dimension_list[:, 0], file_dimension_list[:, 1], "ro")
plt.title("Image sizes")
plt.xlabel("width")
plt.ylabel("height")


# Maximum number of images lie within 500x500 size

# ## Class (Breed) Distribution

# In[ ]:

fig = plt.figure(figsize = (12,5))

ax1 = fig.add_subplot(1,2, 1)
dataset_df.breed.value_counts().tail().plot('bar',ax=ax1,color='gray',title="Breeds with Lowest Counts")

ax2 = fig.add_subplot(1,2, 2)
dataset_df.breed.value_counts().head().plot('bar',ax=ax2,color='black',title="Breeds with Highest Counts")


# ## Understand impact of resizing
# Impact of resizing is pretty evident

# In[ ]:

IMG_INDEX = 100


# In[ ]:

plt.figure(figsize = (10,4))
img = plt.imshow(spi.imread(DATASET_PATH + file_list[IMG_INDEX]))
img.set_cmap('hot')
plt.axis('off')
plt.title("Original Image")


# In[ ]:

scale_size = [160, 80, 40, 20]


# In[ ]:

for scaler in scale_size:
    plt.figure(figsize = (10,4))
    img = plt.imshow(sp.misc.imresize(spi.imread(DATASET_PATH + file_list[IMG_INDEX]), (scaler, scaler)))
    img.set_cmap('hot')
    plt.axis('off')
    plt.title("Scaled Image to {}x{}".format(scaler,scaler))
    plt.show()


# ## Analyze Color Channels

# In[ ]:

dog_image = spi.imread(DATASET_PATH + file_list[IMG_INDEX])


# In[ ]:

dog_r = dog_image.copy() # Red Channel
dog_r[:,:,1] = dog_r[:,:,2] = 0 # set G,B pixels = 0
dog_g = dog_image.copy() # Green Channel
dog_g[:,:,0] = dog_r[:,:,2] = 0 # set R,B pixels = 0
dog_b = dog_image.copy() # Blue Channel
dog_b[:,:,0] = dog_b[:,:,1] = 0 # set R,G pixels = 0

plot_image = np.concatenate((dog_r, dog_g, dog_b), axis=1)
plt.figure(figsize = (10,4))
plt.set_cmap('hot')
plt.axis('off')
plt.imshow(plot_image)


# ## Gray Scale Pixel Distribution

# In[ ]:

from skimage.color import rgb2gray

dog_img_gray = rgb2gray(dog_image)

print('Image shape:', dog_img_gray.shape, '\n')

# 2D pixel map
print('2D image pixel map')
print(np.round(dog_img_gray, 2), '\n')

# flattened pixel feature vector
print('Flattened pixel map:', (np.round(dog_img_gray.flatten(), 2)))


# In[ ]:

fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(1,2, 1)
plt.set_cmap('hot')
plt.axis('off')
ax1.imshow(dog_image, cmap="gray")
ax2 = fig.add_subplot(1,2, 2)
c_freq, c_bins, c_patches = ax2.hist(dog_img_gray.flatten(), bins=30)


# ## Edge Detection

# In[ ]:

from skimage.feature import canny

dog_edges = canny(dog_img_gray, sigma=3)
fig = plt.figure(figsize = (8,4))
plt.set_cmap('hot')
plt.axis('off')
ax2 = fig.add_subplot(1,1, 1)
ax2.imshow(dog_edges, cmap='binary')


# ## Deep Learning based Feature Extraction

# In[ ]:

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K


# ### Setup a very basic 2 Layer CNN

# In[ ]:

model = Sequential()
model.add(Conv2D(4, (4, 4), input_shape=(500, 500, 3), activation='relu', 
                 kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(4, (4, 4), activation='relu', 
                kernel_initializer='glorot_uniform'))


# ### View Model Architecture

# In[ ]:

model.summary()


# ### Hooks to extract features from layers

# In[ ]:

first_conv_layer = K.function([model.layers[0].input, K.learning_phase()], 
                              [model.layers[0].output])
second_conv_layer = K.function([model.layers[0].input, K.learning_phase()], 
                               [model.layers[2].output])


# ### Extract and Visualize Features

# In[ ]:

dog_vec = dog_image.reshape(1, 500, 500, 3)

# extract feaures 
first_conv_features = first_conv_layer([dog_vec])[0][0]
second_conv_features = second_conv_layer([dog_vec])[0][0]

# view feature representations
fig = plt.figure(figsize = (14,4))
ax1 = fig.add_subplot(2,4, 1)
ax1.imshow(first_conv_features[:,:,0])
ax2 = fig.add_subplot(2,4, 2)
ax2.imshow(first_conv_features[:,:,1])
ax3 = fig.add_subplot(2,4, 3)
ax3.imshow(first_conv_features[:,:,2])
ax4 = fig.add_subplot(2,4, 4)
ax4.imshow(first_conv_features[:,:,3])

ax5 = fig.add_subplot(2,4, 5)
ax5.imshow(second_conv_features[:,:,0])
ax6 = fig.add_subplot(2,4, 6)
ax6.imshow(second_conv_features[:,:,1])
ax7 = fig.add_subplot(2,4, 7)
ax7.imshow(second_conv_features[:,:,2])
ax8 = fig.add_subplot(2,4, 8)
ax8.imshow(second_conv_features[:,:,3])

