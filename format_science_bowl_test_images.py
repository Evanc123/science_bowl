import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
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
import glob
import os
import h5py
import shutil
import sklearn
import tempfile
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

num_images = 0
for file in os.listdir("kaggle_test"):
    if file.endswith(".jpg"):
	num_images +=1
files = []
maxPixel = 50
imageSize = maxPixel * maxPixel
num_rows = num_images # one row for each image in the training dataset
num_features = imageSize
X = np.zeros((num_rows, 1, maxPixel, maxPixel), dtype=float)
i = 0
print num_images
for file in os.listdir("kaggle_test"):
    if file.endswith(".jpg"):
	image = imread("kaggle_test/" + file, as_grey = True)
	files.append(file)
	
	image = resize(image,(maxPixel, maxPixel))
	X[i, 0] = image
	i+=1

print X

np.save('X_kaggle_test', X)


dirname = os.path.abspath('./science_bowl/testing')
if not os.path.exists(dirname):
    os.makedirs(dirname)

test_filename = os.path.join(dirname, 'kaggle_test.h5')

with h5py.File(test_filename, 'w') as f:
    f['data'] = X

