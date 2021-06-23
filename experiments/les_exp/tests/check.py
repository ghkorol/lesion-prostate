import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


data = np.load('../preprocessing_out/prosx-0193_img.npy')
data_oscar = np.load('ProstateX-0193_rois.npy')


# print(data.max(), data.min())
print(data.shape)
print(data_oscar.shape)

plt.subplot(1,2,1)
plt.imshow(data[-1].sum(axis=2))

plt.subplot(1,2,2)
plt.imshow(data_oscar[:, :, :, 0].sum(axis=0))

plt.show()

sys.exit()

# for i in range(data.shape[0]):
#     plt.subplot(5,2,i+1)
#     plt.imshow(data.sum(axis=3)[i,:,:])
#     plt.show()

for i in range(data.shape[-1]):
    plt.subplot(6,4,i+1)
    plt.imshow(data[9][:, :, i])
plt.show()
plt.close()




pkl_data = pd.read_pickle('../preprocessing_out/prosx-0193_info.pickle')



print(pkl_data)

