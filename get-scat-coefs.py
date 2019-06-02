import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.restoration import (denoise_wavelet)
from skimage import exposure
from torch.utils import data
import pickle
from torchvision import transforms
from kymatio import Scattering2D
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from torch.autograd import Variable


class newcancer(data.Dataset):
    def __init__(self, parent_dir,samplerate,transform=None,phase = 'trainc',load_prob = True,green = False):
        """
        Args:
            transform (optional): Optional transform to be applied on a sample.
            green: If true, only take the green channel
            samplerate: take how much percent of the original tadaset
            Note: Whenever you change a samplerate, you need to resave the pickle
        """
        self.green = green
        self.transform = transform
        
        if not parent_dir.endswith('/'):
            # Make sure the directory name is correctly given
            parent_dir = parent_dir + '/'

        data_list = []
        for lab in ['/malignant/','/benign/']:
            filelist  = glob.glob(parent_dir +lab+ '**/40X/*.png', recursive=True)
            data_list.extend([(file,0) if lab == '/benign/' else (file,1) for file in filelist]) #include labels before random split
        
        self.data_list = data_list #All the (filename,label)
        
#         self.num_allsamples = len(data_list)
#         self.transform = transform
    
        random.seed(3)
        random.shuffle(data_list)       
        trainlst = data_list[:round(samplerate*len(data_list))]
        testlst = data_list[round(samplerate*len(data_list)):]
       
        
        if phase == 'trainc':
            jpg_list = trainlst
        else:
            jpg_list = testlst
      
        self.jpg_list = jpg_list
    def __len__(self):
        return len(self.image_data_dict)

    def __getitem__(self,index):
        '''
        Return a tuple containing the image tensor and corresponding class for the given index.
        Parameter:
        index: This is the index created by _init_, it's the key of the dict in _init_
               Notice that a single patient could have multiple index associated.
        '''
            
        img = imageio.imread(self.jpg_list[index][0]) #rgb
        tag = self.jpg_list[index][1] #pixle and label    
            
       #isolating green channel:
        if self.green:
            img = img[:,:,1]
            
        img = transforms.ToPILImage()(img)
        img = transforms.functional.resize(img,(60,60))
        
        scattering = Scattering2D(J=2, shape=(60, 60))
        #K = 81*3
    
        if self.transform:
            img = self.transform(img)
        if torch.cuda.is_available():
            img_gpu = img.cuda()
            Simg_gpu = scattering(img)
            return (Simg_gpu,tag)

pdir = '/scratch/jx1047/project/Scattering-colonography/breast'
for rate in [0.1,0.2,0.3]:
    testimg = newcancer(pdir,samplerate = rate,phase = 'testc',transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.8009, 0.6541, 0.7711),(0.11159356, 0.12982282, 0.10046051))]))
    scattest = {}
    for i in range(len(testimg.jpg_list)):
        scattest[i] = testimg.__getitem__(i)
    with open('scat'+str(rate)+'testc'+'.pickle','wb') as handle:
        pickle.dump(scattest,handle)
