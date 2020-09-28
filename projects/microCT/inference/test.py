import os, sys, glob 
import time, itertools
import GPUtil
import time
import h5py
from os import listdir
import scipy.io
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

from connectomics.engine.solver import *
from connectomics.model import *


import os,datetime
import imageio
import h5py
import torch
import torch.nn as nn
import numpy as np
#from threedunet import unet_residual_3d
from torchsummary import summary



#testvolume_name = 'MvsFCSmale4week-3-DownSamp_im.h5'
#pred_name = 'MvsFCSmale4week-3-DownSamp_predictiontest.h5'


def read_h5(filename, dataset=''):
    fid = h5py.File(filename, 'r')
    if dataset == '':
        dataset = list(fid)[0]
    return np.array(fid[dataset])



class Trainer(object):
    r"""Trainer

    Args:
        cfg: YACS configurations.
        device (torch.device): by default all training and inference are conducted on GPUs.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``).
        checkpoint (optional): the checkpoint file to be loaded (default: `None`)
    """
    def __init__(self, cfg, device, mode, checkpoint=None):
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode
        self.checkpoint = checkpoint

        self.model = build_model(self.cfg, self.device)
        #self.update_checkpoint(checkpoint)


    def test(self):

        r"""Testing function.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        volume_loc = self.cfg.INFERENCE.IMAGE_NAME
        output_path = self.cfg.INFERENCE.OUTPUT_PATH
        pred_name = self.cfg.INFERENCE.OUTPUT_NAME

        pred_location = os.path.join(output_path,pred_name)

        model = self.model
        model = nn.DataParallel(model, device_ids=range(self.cfg.SYSTEM.NUM_GPUS))
        model = model.to(device)


        # load pre-trained model
        print('Load pretrained checkpoint: ', self.checkpoint)
        checkpoint = torch.load(self.checkpoint)
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

        
        volume_name = volume_loc
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


        hf1 = h5py.File(pred_location, 'w')
        hf1.create_dataset('dataset1', data=pred_final)
        print("Prediction volume created and saved" , hf1)
        hf1.close()







