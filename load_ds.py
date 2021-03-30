# This is the second step of the data preparation.
# The second step is divided into two stages.
# Each stage has a corresponding program (load_ds.py, load_ds2.py)
# This program combines all of the output from breakdown_2.py into a single dataset.

from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.core.shape_base import vstack

data_path = 'dataset/data/'
labels_path = 'dataset/labels/'

files_data = [f for f in listdir(data_path) if isfile(join(data_path, f))]
files_labels = [f for f in listdir(labels_path) if isfile(join(labels_path, f))]

files_data.sort()
files_labels.sort()

print(len(files_data))
print(len(files_labels))

ds_data = np.vstack([np.load(data_path + f) for f in files_data])
np.save('dataset2/data', ds_data)
ds_labels = np.hstack([np.load(labels_path + f) for f in files_labels])
np.save('dataset2/labels', ds_labels)