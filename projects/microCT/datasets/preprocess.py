import numpy as np
import imageio
import h5py
import os
import pandas as pd 
import numpy as np
import imageio
import h5py

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

base_dir = "path/to/folder"

masks = "ground_truth_file.h5"
volumes = "volume_file.h5"


def read_h5(filename, dataset=''):
    fid = h5py.File(filename, 'r')
    if dataset == '':
        dataset = list(fid)[0]
    return np.array(fid[dataset])

def foreground_crop_threshold(image,label,threshold):
    # only keep the foreground region of the image and label:
    temp_image = np.array(image)
    x_coord, y_coord, z_coord = np.where(temp_image > threshold)
    x_min, x_max = x_coord.min(), x_coord.max()
    y_min, y_max = y_coord.min(), y_coord.max()
    z_min, z_max = z_coord.min(), z_coord.max()
    temp_image = temp_image[z_min:z_max,y_min:y_max,x_min:x_max]
    label = label[z_min:z_max,y_min:y_max,x_min:x_max]
    return temp_image,label

zip_file = zip(volumes, masks)

c = 0

for i, j in zip_file:

    vol = os.path.join(base_dir, i)
    gt = os.path.join(base_dir, j)


    image_volume = read_h5(vol)                                 #reading CT volume 
    gt_volume = read_h5(gt).astype(np.uint16)                   # reading GT


    print(c, "Shape of CT Volume", image_volume.shape)
    print(c, "Shape of Ground truths Label", gt_volume.shape)

    
    image_crop, label_crop = foreground_crop_threshold(image_volume,gt_volume,128)          #Thresholding

    print(c, "Shape of Volume after crop", image_crop.shape)
    print(c, "Shape of Label after crop", label_crop.shape)


    gt_rescaled = resize(label_crop, (112,112,112),order=0, preserve_range=True, anti_aliasing=False)   #resize

    gt_rescaled = gt_rescaled.astype(np.uint16)

    print(c, "Shape of resized Gt", gt_rescaled.shape)

    vol_rescaled = resize(image_crop, (112,112,112), preserve_range=True, anti_aliasing=True)

    print(c, "Shape of resized volume", vol_rescaled.shape)

    vol_rescaled = vol_rescaled.astype(np.uint8)

    hf1 = h5py.File(i, 'w')
    hf1.create_dataset('dataset1', data=vol_rescaled)

    print(c, "Resized volume created and saved", i)
    hf1.close()

    hf3 = h5py.File(j, 'w')
    hf3.create_dataset('dataset2', data=gt_rescaled)
    hf3.close()

    print(c, "Resized GT created and saved", j)



    c=c+1
    


 
