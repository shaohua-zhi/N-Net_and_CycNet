
import torch  ## the top-level pytorch package and tensor library
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from os.path import join as pjoin
from scipy  import misc

import matplotlib.pyplot as plt
import skimage.io as io ## skimage
import torch.utils.data as data
import glob

from torchvision.transforms import transforms as T

from torch.utils.data import Dataset # an abstract class for representing a dataset
from torch.utils.data import DataLoader # wraps a dataset and provides access to the underlying data
import torch.optim as optim
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchvision # from torchvision import transforms  # common transforms for image processing
import numpy as np
import pdb ## the Python debugger


## import N-Net model 
from model_Nnet import *

## import dataset 
from TrainDataset_Nnet import *
     

torch.set_grad_enabled(True)
print(torch.__version__)
print(torchvision.__version__)

BATCH_NUM = 1
WORKER_NUM = 0
EPOCH = 50
LEARNING_RATE = 1e-6;
MINI_BATCH = 1000

## set the index of training  dataset
HMIndex=[ 100, 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 ]
## count the  slice number for each set of dataset
SliceNum = [87, 66, 60, 79, 65, 71, 52, 60, 60, 67, 77, 92, 59, 67, 59, 51, 68 ]

## instance of dataset
x_transform = T.ToTensor()
y_transform = T.ToTensor()


train_dataset_4DCBCT = Nnet_Dataset(
        'E:/Pytorch/4DCBCT_TrainingData/'
		,'E:/Pytorch/4DCBCT_TrainingData/'
		,'E:/Pytorch/4DCBCT_TrainingData/'
        ,HMIndex
        ,SliceNum
        , transform = x_transform
        , target_transform = y_transform
        )

dataloader = DataLoader( train_dataset_4DCBCT, batch_size=1, shuffle=True, num_workers=0 )
#aa, bb, cc = next(iter(dataloader))
print('The total trainingdata has', len(train_dataset_4DCBCT))


# split the dataset into training data and validation data
train_db, val_db = data.random_split(
        train_dataset_4DCBCT
        , [int(len(train_dataset_4DCBCT)*0.9),int(len(train_dataset_4DCBCT)*0.1)])
print('train:', len(train_db), 'validation:', len(val_db))


TrainDataLoader_SpatialCNN = data.DataLoader(
        train_db
        , batch_size = BATCH_NUM
        , shuffle = True
        , num_workers = 0
        )

ValidDataLoader_SpatialCNN = data.DataLoader(
        val_db
        , batch_size = BATCH_NUM
        , shuffle = True
        , num_workers = WORKER_NUM
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Nnet()
model = model.to(device)
        
def adjust_learning_rate(optimizer, epoch,LEARNING_RATE):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LEARNING_RATE * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        



optimizer = optim.Adam( model.parameters(), lr = LEARNING_RATE )

criterion = torch.nn.MSELoss()


MSEtrain = []
MSEval = []

MINI_Epoch = 0

## train loop
for epoch in range(0,EPOCH):
    
    adjust_learning_rate( optimizer, epoch, LEARNING_RATE )
    
    running_loss = 0
    
    for i,batch in enumerate( TrainDataLoader_SpatialCNN, 0 ):
        
        images, prior, labels = batch  # [N, 1, H, W]
        
        images = images.to(device)
        prior = prior.to(device)
        labels = labels.to(device)
       
        optimizer.zero_grad()  
        
        try:
            prediction = model(images,prior)
            
        except RuntimeError as exception:
            if "out of memory" in str( exception ):
                print( "WARNING: out of memory" )
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        loss = criterion( prediction,labels )    # MSE


        loss.backward( retain_graph = True )
        optimizer.step()       
        running_loss += loss.item()

        
        if i % MINI_BATCH == (MINI_BATCH-1):
            print('[epoch %d batch %5d] loss: %.3f' %
                  ( epoch+1, i+1, running_loss)
                    )
            MSEtrain.append([running_loss])
            
    
#            writer_train.add_scalar('Train/SSIMLoss'
#                              , running_loss/MINI_BATCH
#                              , MINI_Epoch ) 
            running_loss=0.0
            
            #### validation            
            running_loss_val = 0

            for j, batch_val in enumerate( ValidDataLoader_SpatialCNN, 0 ):
                
                images_val,prior_val,labels_val = batch_val  # [N, 1, H, W]
                
                images_val = images_val.to(device)
                prior_val = prior_val.to(device)
                labels_val = labels_val.to(device)
                
                prediction_val = model(images_val,prior_val)
        
                loss_val = criterion( prediction_val,labels_val )    # SSIM
                running_loss_val += loss_val.item()
        
        
            print('[epoch %d batch %5d] Val_loss: %.3f' %
                          ( epoch+1, j+1, running_loss_val)
                            )
            MSEval.append([ running_loss_val ])

            MINI_Epoch += 1 
       
    if epoch % 5 == 4:      
        PATH ='./trained_model/Nnet0419_epoch'+str(epoch+1)+'_LR1e-6_17sets_lossMSE.pth'    
        torch.save(model, PATH)  
         
#writer_train.close()   
#writer_val.close()   
numpy_MSEtrain = np.array(MSEtrain)
numpy_MSEval = np.array(MSEval)

# save the MSE value to generate convergence curve
np.save('N-net_MSEtrain.npy',numpy_MSEtrain )
np.save('N-net_MSEval.npy',numpy_MSEval )
