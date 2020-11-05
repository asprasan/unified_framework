'''
------------------------------------------
DEFINE DATALOADER TO FETCH VIDEO SEQUENCES
------------------------------------------
removed recurrence in train data
added return of patches
added hdf5 file access

'''
import torch
from torch.utils import data
import glob
import os
import numpy as np
import scipy.misc
from PIL import Image
import h5py
# import time


class Dataset_load(data.Dataset):
    
    def __init__(self, filepath, dataset, num_samples='all'):
        'Initialization'

        f = h5py.File(filepath, 'r')
        print('\nReading data from %s'%(filepath))
        print('Found', list(f.keys()), '...Reading from', dataset)
        
        if num_samples == 'all':
            self.dset = f[dataset]
        else:
            self.dset = f[dataset][:num_samples]


    def __len__(self):
        'Denotes the total number of samples'

        return self.dset.shape[0]


    def __getitem__(self, index):
        'Generates one sample of data'

        vid = torch.FloatTensor(self.dset[index, ...]) / 255.
        return vid