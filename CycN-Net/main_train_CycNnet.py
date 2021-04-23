import torch  ## the top-level pytorch package and tensor library
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim

# from torchvision import transforms  
# common transforms for image processing
import torchvision 
from torchvision.transforms import transforms as T
import torch.utils.data as data
import matplotlib.pyplot as plt

import numpy as np
import pdb 
import os

## the Python debugger
torch.set_grad_enabled(True)
print( torch.__version__)
print( torchvision.__version__)
print( torch.cuda.device_count())
print( torch.cuda.is_available())

train_BATCH_NUM = 20
val_BATCH_NUM = 1
WORKER_NUM = 8
EPOCH = 50
LEARNING_RATE = 1e-6
MINI_BATCH = 50

## import dataset 
from TrainDataset_CycNnet import *

## import CycN-net model 
from model_CycNnet import *

x_transform = T.ToTensor()
y_transform = T.ToTensor() # normalize image value to [0 1]


## set the index of training  dataset
SliceNum = [ 87, 66, 60, 79, 65, 71, 52, 60, 60, 67, 77, 92, 59, 67, 59, 51, 68 ]

## set the index of training  dataset
HMIndex = [ 100, 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 ]


train_dataset_4DCBCT = TrainDataset_CircleNnet3D(
        './4DCBCT_TrainingData/'
        ,'./4DCBCT_TrainingData/'
		,'./4DCBCT_TrainingData/'
        ,HMIndex
        ,SliceNum
        , transform = x_transform
        , target_transform = y_transform
        )

#dataloader = data.DataLoader(dataset_4DCBCT, batch_size=1, shuffle=True, num_workers=0)
#aa,bb,cc,dd,ee = next(iter(dataloader)) # aa,bb,cc are with size of [1,3,512,512]
    
print('The total trainingdata has', len(train_dataset_4DCBCT))

# split the dataset into training data and validation data
train_db, val_db = data.random_split(
        train_dataset_4DCBCT
        , [int(len(train_dataset_4DCBCT)*0.9),int(len(train_dataset_4DCBCT)*0.1)])
print('train:', len(train_db), 'validation:', len(val_db))


TrainDataLoader_SpatialCNN = data.DataLoader(
        train_db
        , batch_size = train_BATCH_NUM
        , shuffle = True
        , num_workers = WORKER_NUM
        )


ValidDataLoader_SpatialCNN = data.DataLoader(
        val_db
        , batch_size = val_BATCH_NUM
        , shuffle = True
        , num_workers = WORKER_NUM
        )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = CycNnet()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs")
    # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
    model = nn.DataParallel( model, device_ids=[0,1,2] )
    
model.to(device)

## if use the intermediate trained model

#model_PATH ='E:/Pytorch/SpatialTemporalUnet/modelsave/SpatialCNN_BatchNorm_epoch17_LR1e-6_16sets_lossSSIM7.pth'
# PATH ='./modelsave_epoch50_LR1e-4_lossSSIM7.pth'
#model = (torch.load(model_PATH))

        
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
        
        img_seq_1, img_seq_2, img_seq_3, prior, labels = batch  # [N, 1, H, W]

        img_seq_1 = img_seq_1.to(device)
        img_seq_2 = img_seq_2.to(device)
        img_seq_3 = img_seq_3.to(device)

        prior = prior.to(device)
        labels = labels.to(device)
       
        optimizer.zero_grad()  
        
        try:
            prediction = model( img_seq_1, img_seq_2, img_seq_3, prior )
            
        except RuntimeError as exception:
            if "out of memory" in str( exception ):
                print( "WARNING: out of memory" )
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        loss = criterion( prediction,labels )    # MSE


        loss.backward(retain_graph=True)
        optimizer.step()       
        running_loss += loss.item()

	        
        if i % MINI_BATCH == (MINI_BATCH-1):
            print('[epoch %d batch %5d] loss: %.3f' %
                  ( epoch+1, i+1, running_loss)
                    )
            MSEtrain.append([running_loss])
           
            running_loss = 0.0
            
            #### validation            
            running_loss_val = 0

            for j, batch_val in enumerate( ValidDataLoader_SpatialCNN, 0 ):

                images_val_1, images_val_2, images_val_3, prior_val, labels_val = batch_val  # [N, 1, H, W]
                
                images_val_1 = images_val_1.to(device)
                images_val_2 = images_val_2.to(device)
                images_val_3 = images_val_3.to(device)
                prior_val = prior_val.to(device)
                labels_val = labels_val.to(device)

                
                prediction_val = model(images_val_1, images_val_2, images_val_3, prior_val)
                loss_val = criterion( prediction_val,labels_val )  #MSE
                running_loss_val += loss_val.item()
        
        
            print('[epoch %d batch %5d] Val_loss: %.3f' %
                          ( epoch+1, j+1, running_loss_val)
                            )
            MSEval.append([ running_loss_val ])


            MINI_Epoch += 1 
       
    if epoch % 5 == 4:      
        PATH ='./trained_model/CycNnet_epoch'+str(epoch+1)+'_LR1e-6_17sets_lossMSE.pth'    
        torch.save(model, PATH)  
         
  
numpy_MSEtrain = np.array(MSEtrain)
numpy_MSEval = np.array(MSEval)

np.save('CycNnet_MSEtrain.npy',numpy_MSEtrain )
np.save('CycNnet_MSEval.npy',numpy_MSEval )  

#
### Output the intermediate feature map according to the indexs ##
##DownConv1:DownConv1=model(images,prior)[1][0] # 252@8
##DownConv8:DownConv8=model(images,prior)[1][1] # 1@512  
##DownPriorConv1:DownPriorConv1=model(images,prior)[1][2]
##DownPriorConv8:DownPriorConv8=model(images,prior)[1][3]
##UpConv1:UpConv1=model(images,prior)[1][4] # 4@512
##UpConv7:UpConv7=model(images,prior)[1][5] # 512@16   
##FinalConv:FinalConv=model(images,prior)[1][6] 
#
## plt.imshow(FinalConv[0][0].cpu().squeeze().detach().numpy())
#  
### save model
#PATH ='./modelsave_epoch20_LR1e-5_16sets_lossSSIM7.pth'
#state = { 'model': model.state_dict(),
#         'optimizer':optimizer.state_dict(),
#         'epoch': epoch } 
 
#torch.save(model, PATH)

