import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io ## skimage库处理图
import torch.utils.data as data
import glob

from torchvision.transforms import transforms as T


class TrainDataset_CircleNnet3D(data.Dataset):
    
    def __init__( self, root_FDKImg, root_Prior, root_GT, HMIndex, SliceNum, transform = None, target_transform = None):       
            
        self.root_FDKImg = root_FDKImg
        self.root_Prior = root_Prior
        self.root_GT = root_GT
        
        SeriesNum = len(HMIndex)
        PhaseSequence_5frames = [[9,10,1,2,3],[10,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10],[7,8,9,10,1],[8,9,10,1,2]]
        TrainingSet = []
        
#        SliceNum = [66,60,79,65,71,52,60,60,67,77,92,59,67,59,51,68]
#        HMIndex=[ 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 ]
    
        for Series in range(0,SeriesNum): 
            
            for phase in range(0,10):
                
                for sliceindex in range( 1, SliceNum[Series]+1 ):
                    
                    new_degraded_1 = glob.glob( root_FDKImg + str(HMIndex[Series])+'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][0])+'/Recons'+str(sliceindex)+'.png') 
                    new_degraded_2 = glob.glob( root_FDKImg + str(HMIndex[Series])+'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][1])+'/Recons'+str(sliceindex)+'.png')    
                    new_degraded_3 = glob.glob( root_FDKImg + str(HMIndex[Series])+'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][2])+'/Recons'+str(sliceindex)+'.png')
                    new_degraded_4 = glob.glob( root_FDKImg + str(HMIndex[Series])+'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][3])+'/Recons'+str(sliceindex)+'.png')    
                    new_degraded_5 = glob.glob( root_FDKImg + str(HMIndex[Series])+'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][4])+'/Recons'+str(sliceindex)+'.png')

                    new_prior = glob.glob( root_Prior + str(HMIndex[Series]) + 'HM10395/Prior_ArtifactFree/Prior'+str(sliceindex)+'.png')     
                    new_mask = glob.glob( root_GT + str(HMIndex[Series])+'HM10395/GT_Phase'+str(PhaseSequence_5frames[phase][2])+'/GT'+str(sliceindex)+'.png')
                
                    aa = [ new_degraded_1, new_degraded_2, new_degraded_3,new_degraded_4,new_degraded_5 ], [ new_prior ], [ new_mask ]
                    TrainingSet.append(aa) 

            
        self.TrainingSet = TrainingSet
        self.transform = transform
        self.target_transform = target_transform
        
# to get an element from the dataset at a specific index location with the dataset     
    def __getitem__(self,index):  
        
        img_1 = np.zeros(( 1, 512, 512 ))
        img_2 = np.zeros(( 1, 512, 512 ))
        img_3 = np.zeros(( 1, 512, 512 ))
        img_4 = np.zeros(( 1, 512, 512 ))
        img_5 = np.zeros(( 1, 512, 512 ))
        
        img1_noise_seq = np.zeros(( 3, 512, 512 ))
        img2_noise_seq = np.zeros(( 3, 512, 512 ))  
        img3_noise_seq = np.zeros(( 3, 512, 512 ))
        
        img_prior_seq = np.zeros((1, 512, 512))
        img_gt_seq = np.zeros(( 1, 512, 512 )) 
        
#        for kk in range(0,3):

#            img_x = io.imread(self.TrainingSet[index][0][kk:kk+2][0])
#            img1_noise_seq[kk] = np.array(img_x)
        img_1 = io.imread(self.TrainingSet[index][0][0][0])
        img_2 = io.imread(self.TrainingSet[index][0][1][0])
        img_3 = io.imread(self.TrainingSet[index][0][2][0])
        img_4 = io.imread(self.TrainingSet[index][0][3][0])
        img_5 = io.imread(self.TrainingSet[index][0][4][0])

        img1_noise_seq[0]=np.array(img_1)
        img1_noise_seq[1]=np.array(img_2)
        img1_noise_seq[2]=np.array(img_3)
        
        img2_noise_seq[0]=np.array(img_2)
        img2_noise_seq[1]=np.array(img_3)
        img2_noise_seq[2]=np.array(img_4)
        
        img3_noise_seq[0]=np.array(img_3)
        img3_noise_seq[1]=np.array(img_4)
        img3_noise_seq[2]=np.array(img_5)
        
#        img1_noise_seq = img_seq_5frames
#        img2_noise_seq = io.imread(self.TrainingSet[index][0][1:3])
#        img3_noise_seq = io.imread(self.TrainingSet[index][0][2:4])
        
        img_prior = io.imread(self.TrainingSet[index][1][0][0])
        img_prior_seq[0] = np.array(img_prior)
       
      
        img_gt = io.imread(self.TrainingSet[index][2][0][0])
        img_gt_seq[0] = np.array(img_gt)
        
#        img_residue= img_x_prior_seq[1]-img_gt_seq[0]
#        img_residue_seq[0] = np.array(img_residue)
#        img_seq_5frames= img_seq_5frames.transpose((1,2,0))   
        img1_noise_seq = img1_noise_seq.transpose((1,2,0))   
        img2_noise_seq = img2_noise_seq.transpose((1,2,0))  
        img3_noise_seq = img3_noise_seq.transpose((1,2,0))  
        img_prior_seq = img_prior_seq.transpose((1,2,0))  
        img_gt_seq = img_gt_seq.transpose((1,2,0))


        if self.transform is not None:
#            img_seq_5frames = self.transform(img_seq_5frames).float().div(255)
            img1_noise_seq = self.transform(img1_noise_seq).float().div(255).unsqueeze(dim=0)
            img2_noise_seq = self.transform(img2_noise_seq).float().div(255).unsqueeze(dim=0)
            img3_noise_seq = self.transform(img3_noise_seq).float().div(255).unsqueeze(dim=0)
            img_prior_seq = self.transform(img_prior_seq).float().div(255)


        if self.target_transform is not None:
            img_gt_seq = self.target_transform(img_gt_seq).float().div(255)
            
        return img1_noise_seq, img2_noise_seq,img3_noise_seq,img_prior_seq, img_gt_seq #返回的是图片，或者tensor
#        return img_seq_5frames, img_prior_seq, img_gt_seq    
    
    def __len__(self):   # retures the length of the dataset
        return len(self.TrainingSet) 
    
    
x_transform = T.ToTensor()
y_transform = T.ToTensor()
#            
  

dataset_4DCBCT = TrainDataset_CircleNnet3D(
        './4DCBCT_TrainingData/'
		,'./4DCBCT_TrainingData/'
        ,'./4DCBCT_TrainingData/'
        ,[101]
        ,[66]
        , transform = x_transform
        , target_transform = y_transform
        )

dataloader = data.DataLoader(dataset_4DCBCT, batch_size=1, shuffle=True, num_workers=0)

aa,bb,cc,dd,ee = next(iter(dataloader)) 
##
#
#plt.subplot(3,3,1)
#plt.imshow(aa[0][0][0])
#plt.subplot(3,3,2)
#plt.imshow(aa[0][0][1])
#plt.subplot(3,3,3)
#plt.imshow(aa[0][0][2])     
#
#
#
#plt.subplot(3,3,4)
#plt.imshow(bb[0][0][0])
#plt.subplot(3,3,5)
#plt.imshow(bb[0][0][1])
#plt.subplot(3,3,6)
#plt.imshow(bb[0][0][2]) 
#
#plt.subplot(3,3,7)
#plt.imshow(cc[0][0][0])
#plt.subplot(3,3,8)
#plt.imshow(cc[0][0][1])
#plt.subplot(3,3,9)
#plt.imshow(cc[0][0][2]) 