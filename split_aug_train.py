import h5py
import os.path as path
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
import scipy
from skimage.feature import peak_local_max
import h5py
import os.path as path
dirname = '/home/ecater/Desktop/caffe/examples/science_bowl/data/'
n = 4
aug_train = h5py.File(dirname + 'aug_train.h5', 'r')
X = aug_train['data']
y = aug_train['label']
index = X.shape[0] / n

comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
for i in range(n):
    train_filename = os.path.join(dirname, 'aug_train_%s.h5' % i)
    print train_filename
    print X[i * index: i * index + index].shape
    with h5py.File(train_filename, 'w') as f:
        f.create_dataset('data', data=X[i * index: i * index + index], **comp_kwargs)
        f.create_dataset('label', data=y[i * index: i * index + index].astype(np.float32), **comp_kwargs)
    with open(os.path.join(dirname, 'aug_train_%s.txt' % i), 'w') as f:
        f.write(train_filename + '\n')
    print "done with %s" % i


aug_train.close()

# make 4 seperate 'aug_train_n.h5' files
"""
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(train_filename, 'w') as f:
    f.create_dataset('data', data=un_X, **comp_kwargs)
    f.create_dataset('label', data=y.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'aug_train.txt'), 'w') as f:
    f.write(train_filename + '\n')




train_filename = os.path.join(dirname, 'aug_train.h5')
test_filename = os.path.join(dirname, 'aug_test.h5')"""

