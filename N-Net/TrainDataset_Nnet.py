import os
import torch
from os.path import join as pjoin
from scipy  import misc

import matplotlib.pyplot as plt
import skimage.io as io ## skimage
import torch.utils.data as data
import glob

from torchvision.transforms import transforms as T
import torch.utils.data as data
from torch.utils.data import Dataset # an abstract class for representing a dataset
from torch.utils.data import DataLoader # wraps a dataset and provides access to the underlying data

## Training Dataset 
class Nnet_Dataset(data.Dataset):
    
    def __init__( self, root_FDKImg, root_Prior, root_GT, HMIndex, SliceNum, transform = None, target_transform = None):
        self.root_FDKImg = root_FDKImg
        self.root_Prior = root_Prior
        self.root_GT = root_GT
        TrainingSet = []

        for SetIndex in range(0, len(HMIndex)):
        
            for phase in range(1,11):

                for sliceindex in range(0,SliceNum[SetIndex]):

                    new_degraded = glob.glob( root_FDKImg + str( HMIndex[SetIndex] )+'HM10395/DegradePhase'+str(phase)+'/Recons'+str(sliceindex+1)+'.png')
                    new_prior = glob.glob( root_Prior + str( HMIndex[SetIndex] )+'HM10395/Prior/Prior'+str(sliceindex+1)+'.png')     
                    new_mask = glob.glob( root_GT + str( HMIndex[SetIndex] )+'HM10395/GT_Phase'+str(phase)+'/GT'+str(sliceindex+1)+'.png')

                    TrainingSet.append([new_degraded, new_prior, new_mask]) 

            self.TrainingSet = TrainingSet
            self.transform = transform
            self.target_transform = target_transform
        
# to get an element from the dataset at a specific index location with the dataset     
    def __getitem__(self,index):  #load data，return[image,label]

        x_path,prior_path,gt_path = self.TrainingSet[index]
        img_x = io.imread(x_path[0]) # image demension 512*512，value scale[0 255]
        img_prior = io.imread(prior_path[0])
        img_gt = io.imread(gt_path[0])

        if self.transform is not None:
            img_x = self.transform(img_x)
            img_prior = self.transform(img_prior)
        if self.target_transform is not None:
            img_gt = self.target_transform(img_gt)
            
        return img_x, img_prior, img_gt 
    
    
    def __len__(self):   # retures the length of the dataset
        return len(self.TrainingSet) 
    

x_transform = T.ToTensor()
y_transform = T.ToTensor()
#        
SliceNum = [ 66, 60, 79, 65, 71, 52, 60, 60, 67, 77, 92, 59, 67, 59, 51, 68 ]
HMIndex = [ 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 ]
#
dataset_4DCBCT = Nnet_Dataset(
        '/home/zsh/DL/4DCBCT_TrainingData/'
		,'/home/zsh/DL/4DCBCT_TrainingData/'
		,'/home/zsh/DL/4DCBCT_TrainingData/'
        ,HMIndex
        ,SliceNum
        , transform = x_transform
        , target_transform = y_transform
        )

dataloader = DataLoader( dataset_4DCBCT, batch_size=1, shuffle=True, num_workers=0 )
#
aa, bb, cc = next(iter(dataloader))
#
#print(aa.shape)
#
#
#plt.figure()
#plt.subplot(1,3,1)
#plt.imshow(aa[0][0],cmap='gray')
#plt.subplot(1,3,2)
#plt.imshow(bb[0][0],cmap='gray')
#plt.subplot(1,3,3)
#plt.imshow(cc[0][0],cmap='gray')
#plt.show()