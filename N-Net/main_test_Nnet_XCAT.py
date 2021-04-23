## test XCAT female 512
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
import matplotlib.pyplot as plt
from PIL import Image

from os.path import join as pjoin
from scipy  import misc

import skimage.io as io ## skimage 
import torch.utils.data as data
import glob


## import N-Net model 
from model_Nnet import *

## import dataset 
from TestingDataset_Nnet_XCAT import *


######### import trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_PATH ='./trained_model/Nnet_epoch50_LR1e-6_17sets_lossMSE.pth'
model_save = ( torch.load( model_PATH ) )

## set the save path
root_save = r'./Results/XCATfemale512/'

######### import dataset  test


x_transform = T.ToTensor()
y_transform = T.ToTensor()


SliceNum = 160

for SetIndex in range(0, 1):
    
    test_dataset_4DCBCT = Test_XCATdataset(
            './4DCBCT_TestingData/XCAT_female_512_slice/'  
			,'./4DCBCT_TestingData/XCAT_female_512_slice/'
			,'./4DCBCT_TestingData/XCAT_female_512_slice/'			
            , SliceNum
            , transform = x_transform
            , target_transform = y_transform
            )
    
    test_dataloader = DataLoader(
            test_dataset_4DCBCT
            , batch_size = 1
            , shuffle = False
            , num_workers = 0
            )

    for i, batch in enumerate( test_dataloader, 0 ):
        
        # [N, 1, H, W]
        images, prior = batch  
            
        images = images.to(device)
        prior = prior.to(device)

        
        PhaseIndex, SliceIndex = divmod( i, SliceNum )
        
        path = root_save+'/CNNPhase'+str(PhaseIndex+1)

        isExists = os.path.exists(path)

        if not isExists:

            os.makedirs(path)
            print( path + 'create successfully!')
        
        with torch.no_grad():
            
            outputs = model_save(images, prior)

        output = outputs[0][0].mul(255).cpu().detach().squeeze().numpy()
        heatmap = np.uint8( np.interp( output, (output.min(), output.max()), (0, 255)))
        im = Image.fromarray(heatmap)
        im.save(root_save + '/CNNPhase' + str(PhaseIndex+1) + '/Processed' + str(SliceIndex+1) +'.png')


print('Print the processed image by N-Net...')
outputs.shape
print(outputs[0][0][200][300])
plt.subplot(1,3,1)
plt.imshow(images[0][0].cpu().detach().numpy())
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(prior[0][0].cpu().detach().numpy())
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(outputs[0][0].cpu().detach().numpy())
plt.axis('off')