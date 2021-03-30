import numpy as np

ds_data = np.load('dataset3/data.npy', allow_pickle=True)
ds_labels = np.load('dataset3/labels.npy', allow_pickle=True)
print(ds_data.shape)
print(ds_labels.shape)

count_0 = 0
count_1 = 1
for i in ds_labels:
    if i == 0.0:
        count_0 += 1
    if i == 1.0:
        count_1 += 1

print(f'0: {count_0}')
print(f'1: {count_1}')