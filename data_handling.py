import numpy as np
#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import sklearn
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
import h5py

directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 50
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 1 # for our ratio

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows * 4, 1, maxPixel, maxPixel), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows * 4))

files = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()
print folder
print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            #axisratio = getMinorMajorRatio(image)
            for j in range(4):
                image = resize(np.rot90(image, j), (maxPixel, maxPixel))
            
            # Store the rescaled image pixels and the axis ratio
                X[i, 0] = image
            #X[i, imageSize] = axisratio
            
            # Store the classlabel
                y[i] = label
                i += 1
            # report progress for each 5% done  
                report = [int((j+1)*num_rows/20.) for j in range(20)]
                if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    label += 1

np.save("X_aug_all.npy", X)
np.save("y_aug_all.npy", y)

X = np.load("X_aug_all.npy")
y = np.load("y_aug_all.npy")
flatten_X = np.zeros((X.shape[0], 2500))
for i in range(X.shape[0]):
    flatten_X[i] = X[i].flatten()             #Flatten for cross val syntax 
f_X, f_Xt, y, yt = sklearn.cross_validation.train_test_split(flatten_X, y)
print f_X.shape, f_Xt.shape, y.shape, yt.shape

un_X = np.zeros((f_X.shape[0], 1, 50, 50))
un_Xt = np.zeros((f_Xt.shape[0], 1, 50, 50))
for i in range(f_X.shape[0]):
    un_X[i] = f_X[i].reshape(1, 50, 50)
for i in range(f_Xt.shape[0]):
    un_Xt[i] = f_Xt[i].reshape(1, 50, 50)
np.save("un_X.npy", un_X)
np.save("un_Xt.npy", un_Xt)
print un_Xt.shape, un_X.shape
    


dirname = os.path.abspath('./hdf5_classification/aug/data')
if not os.path.exists(dirname):
    os.makedirs(dirname)

train_filename = os.path.join(dirname, 'aug_train.h5')
test_filename = os.path.join(dirname, 'aug_test.h5')

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
with h5py.File(train_filename, 'w') as f:
    f['data'] = un_X
    f['label'] = y.astype(np.float32)
with open(os.path.join(dirname, 'aug_train.txt'), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')
    
# HDF5 is pretty efficient, but can be further compressed.
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=un_Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'aug_test.txt'), 'w') as f:
    f.write(test_filename + '\n')

