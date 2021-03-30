import numpy as np
from sklearn.model_selection import train_test_split

ds_data = np.load('dataset2/data.npy', allow_pickle=True)
ds_labels = np.load('dataset2/labels.npy', allow_pickle=True)
print(ds_data.shape)
print(ds_labels.shape)
x_train, x_val, y_train, y_val = train_test_split(ds_data, ds_labels, train_size=0.012)
np.save(f'dataset3/data', x_train)
np.save(f'dataset3/labels', y_train)