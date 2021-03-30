# This program is simply used to check 
# the contents of the resulting dataset.

import numpy as np

ds_data = np.load('dataset3/data.npy', allow_pickle=True)
ds_labels = np.load('dataset3/labels.npy', allow_pickle=True)
print(ds_data.shape)
print(ds_labels.shape)

count_0 = 0
count_1 = 1
index_0 = -1
for i in ds_labels:
    if i == 0.0:
        count_0 += 1
        if index_0 == -1:
            index_0 = count_0 + count_1 - 1
    if i == 1.0:
        count_1 += 1

print(f'0: {count_0}')
print(f'1: {count_1}')
print(f'i: {index_0}')

print(ds_data[821])
print(ds_labels[821])

print(ds_data[0])
print(ds_labels[0])