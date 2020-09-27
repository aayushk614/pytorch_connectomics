import os,datetime
import imageio
import h5py
import torch
import torch.nn as nn
import numpy as np
from threedunet import unet_residual_3d
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_h5(filename, dataset=''):
    fid = h5py.File(filename, 'r')
    if dataset == '':
        dataset = list(fid)[0]
    return np.array(fid[dataset])


model= unet_residual_3d(in_channel=1, out_channel=13).to(device)
model = nn.DataParallel(model, device_ids=range(4))
model = model.to(device)

checkpoint = 'checkpoint_50000.pth.tar'

# load pre-trained model
print('Load pretrained checkpoint: ', checkpoint)
checkpoint = torch.load(checkpoint)
print('checkpoints: ', checkpoint.keys())

# update model weights
if 'state_dict' in checkpoint.keys():
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.module.state_dict() # nn.DataParallel
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict 
    model_dict.update(pretrained_dict)    
    # 3. load the new state dict
    model.module.load_state_dict(model_dict) # nn.DataParallel
    
    print("new state dict loaded ")
    
model.eval()

volume_name = 'MvsFCSmale4week-3-DownSamp_im.h5'
image_volume = read_h5(volume_name)   #reading CT volume 
vol = image_volume

volume = torch.from_numpy(vol).to(device, dtype=torch.float)
volume = volume.unsqueeze(0)
volume = volume.unsqueeze(0)

pred = model(volume)
pred = pred.squeeze(0)

pred = pred.cpu()

pred_final = np.argmax(pred.detach().numpy(),axis=0).astype(np.uint16)
print("Shape of Predictions after argmax() function ", pred_final.shape)


hf1 = h5py.File('MvsFCSmale4week-3-DownSamp_pred_im.h5', 'w')
hf1.create_dataset('dataset1', data=pred_final)
print("Prediction volume created and saved" , hf1)
hf1.close()
