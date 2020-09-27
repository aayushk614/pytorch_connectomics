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

from .solver import *
from connectomics.model import *


def seg_to_targets(label, topts):
    out = [None]*len(topts)
    for tid,topt in enumerate(topts):
        if topt == '-1':
            out[tid] = label[None,:].astype(np.float32)

    return out

base_dir = "CT_data/dsample"       # resampled volumes location (112,112,112)

class Cthousefly(Dataset):  
    def __init__(self, root_dir, transform=None): 
        self.root_dir = root_dir
        xl = pd.read_csv('/n/home03/aayushk614/aayush/exp2_new_mito/pytorch_connectomics/CT_data/train.csv')
        self.trainfile = xl['Volumes']
        self.groundtruths = xl['GT']
        self.transform = transform
        self.iter_num = 100000


    def __len__(self):
        return self.iter_num
    
    def __getitem__(self,idx):
        
        idx = idx % len(self.groundtruths)
        volume_name=os.path.join(self.root_dir , self.trainfile[idx])
        gt_name = os.path.join(self.root_dir, self.groundtruths[idx])

        fid_v = h5py.File(volume_name, 'r')
        dataset_v = list(fid_v)[0]

        fid_gt = h5py.File(gt_name, 'r')
        dataset_gt = list(fid_gt)[0]

        volume = np.array(fid_v[dataset_v])

        ground_truth = np.array(fid_gt[dataset_gt])

        ground_truth = seg_to_targets(ground_truth, ['-1']) 

        if self.transform:
            volume = self.transform(volume)

        return volume, ground_truth

trainset = Cthousefly(root_dir="/n/home03/aayushk614/aayush/exp2_new_mito/pytorch_connectomics/CT_data/dsample")

dataloader_simple = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=4)

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

        self.model = build_model(self.cfg, self.device)
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
        if checkpoint is not None:
            self.update_checkpoint(checkpoint)

        self.dataloader = iter(dataloader_simple)

        self.monitor = build_monitor(self.cfg)
        self.criterion = nn.CrossEntropyLoss()

        # add config details to tensorboard
        self.monitor.load_config(self.cfg)
        self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
        self.inference_output_name = self.cfg.INFERENCE.OUTPUT_NAME


    def train(self):
        r"""Training function.
        """
        # setup
        self.model.train()
        self.monitor.reset()
        self.optimizer.zero_grad()

        for iteration in range(self.total_iter_nums):
            iter_total = self.start_iter + iteration
            start = time.perf_counter()
            # load data
            batch = next(self.dataloader)
            volume, target = batch
            time1 = time.perf_counter()

            target_vis = target
                        
            volume = volume.to(self.device, dtype=torch.float)
            volume = volume.unsqueeze(1)

            target = target[0].to(self.device, dtype=torch.long)
            target = np.squeeze(target, axis=1)

            pred = self.model(volume)

            pred_vis = pred.argmax(1)
            pred_vis = pred_vis.unsqueeze(0).to(self.device, dtype=torch.float)
    
            loss = self.criterion(pred, target)


            # compute gradient
            loss.backward()
            if (iteration+1) % self.cfg.SOLVER.ITERATION_STEP == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # logging and update record
            do_vis = self.monitor.update(self.lr_scheduler, iter_total, loss, self.optimizer.param_groups[0]['lr']) 
        
            if do_vis:


                self.monitor.visualize(self.cfg, volume, target_vis, pred_vis, iter_total)
                # Display GPU stats using the GPUtil package.
                GPUtil.showUtilization(all=True)
            

            # Save model
            if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                self.save_checkpoint(iter_total)

            # update learning rate
            self.lr_scheduler.step(loss) if self.cfg.SOLVER.LR_SCHEDULER_NAME == 'ReduceLROnPlateau' else self.lr_scheduler.step()

            end = time.perf_counter()
            print('[Iteration %05d] Data time: %.5f, Iter time:  %.5f' % (iter_total, time1 - start, end - start))

            # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to 
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
            del loss, pred


    # -----------------------------------------------------------------------------
    # Misc functions
    # -----------------------------------------------------------------------------
    def save_checkpoint(self, iteration):
        state = {'iteration': iteration + 1,
                 'state_dict': self.model.module.state_dict(), # Saving torch.nn.DataParallel Models
                 'optimizer': self.optimizer.state_dict(),
                 'lr_scheduler': self.lr_scheduler.state_dict()}
                 
        # Saves checkpoint to experiment directory
        filename = 'checkpoint_%05d.pth.tar' % (iteration + 1)
        filename = os.path.join(self.output_dir, filename)
        torch.save(state, filename)

    def update_checkpoint(self, checkpoint):
        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint)
        print('checkpoints: ', checkpoint.keys())
        
        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            model_dict = self.model.module.state_dict() # nn.DataParallel
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict 
            model_dict.update(pretrained_dict)    
            # 3. load the new state dict
            self.model.module.load_state_dict(model_dict) # nn.DataParallel   

        if not self.cfg.SOLVER.ITERATION_RESTART:
            # update optimizer
            if 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # update lr scheduler
            if 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # load iteration
            if 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']


