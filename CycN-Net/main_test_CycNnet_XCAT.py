######## test XCATfemale512

import torch  ## the top-level pytorch package and tensor library
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

# from torchvision import transforms  # common transforms for image processing
from torchvision.transforms import transforms as T

from torch.utils.data import Dataset # an abstract class for representing a dataset
from torch.utils.data import DataLoader # wraps a dataset and provides access to the underlying data

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import os
import torch
from os.path import join as pjoin
from scipy  import misc

import skimage.io as io ## skimage
import torch.utils.data as data
import glob

from torchvision.transforms import transforms as T
import torch.utils.data as data
from torch.utils.data import Dataset # an abstract class for representing a dataset
from torch.utils.data import DataLoader # wraps a dataset and provides access to the underlying data

## import CycN-Net model 
from model_CycNnet import *

## import testing data
from TestingDataset_CycNnet_XCAT import *

   
x_transform = T.ToTensor()
y_transform = T.ToTensor()

SliceNum = 160

dataset_4DCBCT = TestingDataset_XCAT(
        './4DCBCT_TestingData/XCAT_female_512_slice/'
        ,'./4DCBCT_TestingData/XCAT_female_512_slice/'
		,'./4DCBCT_TestingData/XCAT_female_512_slice/'
        ,SliceNum
        ,transform = x_transform
        ,target_transform = y_transform
        )

test_dataloader = DataLoader( dataset_4DCBCT, batch_size=1, shuffle=False, num_workers=0 )

## load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_PATH ='./trained_model/CycNnet_epoch50_LR1e-6_17sets_lossMSE.pth'    
model_save = torch.load(model_PATH, map_location='cuda:0')
if isinstance(model_save,torch.nn.DataParallel):
		model_save = model_save.module   

# set the save path of processed image        
root_save = r'./Results/CycNnet_Result_XCATfemale/'

for i, batch in enumerate( test_dataloader, 0 ):
    
        img_seq_1, img_seq_2, img_seq_3, prior, labels = batch
        
        img_seq_1 = img_seq_1.to(device)
        img_seq_2 = img_seq_2.to(device)
        img_seq_3 = img_seq_3.to(device)
        prior = prior.to(device)
        labels = labels.to(device)
        
        PhaseIndex, SliceIndex = divmod( i, SliceNum )
        
        path = root_save+'/Phase'+str( PhaseIndex + 1 )

        isExists = os.path.exists(path)

        if not isExists:

            os.makedirs(path)
            print( path + 'create successfully!')
        
        with torch.no_grad():
            
            output = model_save(img_seq_1,img_seq_2,img_seq_3, prior)
            
        output = output[0][0].mul(255).cpu().detach().squeeze().numpy()
        heatmap = np.uint8( np.interp( output, (output.min(), output.max()), (0, 255)))
        im = Image.fromarray(heatmap)
        im.save( path + '/Processed' + str(SliceIndex+1) +'.png')