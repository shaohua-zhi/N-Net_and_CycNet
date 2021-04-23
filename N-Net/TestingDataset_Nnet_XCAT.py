## Testing Dataset XCAT phantom

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


class Test_XCATdataset(data.Dataset):
    
    def __init__( self, root_FDKImg, root_Prior, root_GT, SliceNum, transform = None, target_transform = None):
        
        self.root_FDKImg = root_FDKImg
        self.root_Prior = root_Prior
        self.root_GT = root_GT
#        seeds=[ 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 ]
        TrainingSet = []
#        sliceNum = [66,60,79,65,71,52,60,60,67,77,92,59,67,59,51,68]
       
#        for i,seeds_index in enumerate(seeds,0):      
      
        for phase in range(1,11):
            
            for sliceindex in range(1,SliceNum+1):
                       
                new_degraded = glob.glob( root_FDKImg +'/Degraded/Phase'+str(phase)+'/Degraded'+str(sliceindex)+'.png')
                new_prior = glob.glob( root_Prior + '/ArtifactReduce_Prior/Prior'+str(sliceindex)+'.png')     
                       
                TrainingSet.append([new_degraded, new_prior]) 
           
        self.TrainingSet = TrainingSet
        self.transform = transform
        self.target_transform = target_transform
        
# to get an element from the dataset at a specific index location with the dataset     
    def __getitem__(self,index): 

        x_path, prior_path = self.TrainingSet[index]
        img_x = io.imread(x_path[0]) 
        img_prior = io.imread(prior_path[0])


        if self.transform is not None:
            img_x = self.transform(img_x)
            img_prior = self.transform(img_prior)
#        if self.target_transform is not None:
#            img_gt = self.target_transform(img_gt)
            
        return img_x, img_prior 
    
    
    def __len__(self):   # retures the length of the dataset
        return len(self.TrainingSet) 