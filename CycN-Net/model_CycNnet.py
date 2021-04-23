import torch  ## the top-level pytorch package and tensor library
import torch.nn as nn
import torch.nn.functional as F


class CycNnet(nn.Module):

    def ExtractionBlock_3D( self, InChannel_1, OutChannel_1, Kernal_1_depth, Kernal_1 ):
        DownBlock_3D = nn.Sequential(
                nn.Conv3d( InChannel_1, OutChannel_1, ( Kernal_1_depth, Kernal_1, Kernal_1 )),
                nn.PReLU()
#                nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2 ),
#                nn.PReLU()
                )
        return DownBlock_3D
    
    def ExtractionBlock_3D_2( self, InChannel_2, OutChannel_2, Kernal_2 ):
        DownBlock_3D = nn.Sequential(
#                nn.Conv3d( InChannel_1, OutChannel_1, ( Kernal_1_depth, Kernal_1, Kernal_1 )),
#                nn.PReLU(),
                nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2 ),
                nn.PReLU()
                )
        return DownBlock_3D
    
    def ExtractionBlock( self, InChannel_1, OutChannel_1, Kernal_1, InChannel_2, OutChannel_2, Kernal_2 ):
        DownBlock = nn.Sequential(
                nn.Conv2d( InChannel_1, OutChannel_1, Kernal_1 ),
                nn.PReLU(),
                nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2 ),
                nn.PReLU()
                ) 
        return DownBlock
       
    
    def ExpansionBlock(self,InChannel1, OutChannel1, Kernel1, InChannel2, OutChannel2, Kernel2):
        UpBlock = nn.Sequential(
            nn.ConvTranspose2d(in_channels= InChannel1, out_channels=OutChannel1, kernel_size= Kernel1 )
            ,nn.PReLU()
            ,nn.ConvTranspose2d(in_channels=InChannel2 , out_channels=OutChannel2, kernel_size= Kernel2 )
            ,nn.PReLU()
            )
        return UpBlock
    
    
    def FinalConv(self, InChannel_1, OutChannel_1, Kernal_1, InChannel_2, OutChannel_2, Kernal_2, InChannel_3, OutChannel_3, Kernal_3):
        finalconv = nn.Sequential(
            nn.Conv2d( InChannel_1, OutChannel_1, Kernal_1, padding=0 )
            ,nn.BatchNorm2d(OutChannel_1,track_running_stats=False) 
            ,nn.PReLU()
            ,nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2, padding=1 )
            ,nn.BatchNorm2d(OutChannel_2,track_running_stats=False)
            ,nn.PReLU()
            ,nn.Conv2d( InChannel_3, OutChannel_3, Kernal_3, padding=0 )
#            ,nn.BatchNorm2d(OutChannel_3,track_running_stats=False)
            ,nn.PReLU()
            )
        return finalconv
   
    
    def __init__(self):
        super(CycNnet, self).__init__()        

        ## Encode
        self.conv_encode1 = self.ExtractionBlock_3D( 1, 4, 3, 5 ) ## (512-4-4)=504
        self.conv_encode1_2 = self.ExtractionBlock_3D_2( 4, 4, 5 ) ## (512-4-4)=504

        self.pool1 = nn.MaxPool2d(2) #504/2=252
        self.conv_encode2 = self.ExtractionBlock( 4, 8, 3, 8, 8, 3 ) ## (252-2-2)=248
        self.pool2 = nn.MaxPool2d(2) #248/2=124
        self.conv_encode3 = self.ExtractionBlock( 8, 16, 3, 16, 16, 3 ) ## (124-2-2)=120
        self.pool3 = nn.MaxPool2d(2) # 120/2=60
        self.conv_encode4 = self.ExtractionBlock( 16, 32, 3, 32, 32, 3 ) ## (60-2-2)=56
        self.pool4 = nn.MaxPool2d(2) ## 56/2=28
        self.conv_encode5 = self.ExtractionBlock( 32, 64, 3, 64, 64, 3 ) ## (28-2-2)=24
        self.pool5 = nn.MaxPool2d(2) ## 24/2=12
        self.conv_encode6 = self.ExtractionBlock( 64, 128, 3, 128, 128, 3 ) ## （12-2-2）=8 
        self.pool6 = nn.MaxPool2d(2) ## 8/2=4
        self.conv_encode7 = self.ExtractionBlock( 128, 256, 3, 256, 256, 1 ) ## (4-2-0)=2
        self.pool7 = nn.MaxPool2d(2) ## 2/2=1
        self.conv_encode8= self.ExtractionBlock( 256, 256, 1, 256, 256, 1 ) ## 1


        self.conv_encode1_prior = self.ExtractionBlock( 1, 4, 5, 4, 4, 5 ) ## (512-4-4)=504
        self.pool1_prior = nn.MaxPool2d(2) #504/2=252
        self.conv_encode2_prior = self.ExtractionBlock( 4, 8, 3, 8, 8, 3 ) ## (252-2-2)=248
        self.pool2_prior = nn.MaxPool2d(2) #248/2=124
        self.conv_encode3_prior = self.ExtractionBlock( 8, 16, 3, 16, 16, 3 ) ## (124-2-2)=120
        self.pool3_prior = nn.MaxPool2d(2) # 120/2=60
        self.conv_encode4_prior = self.ExtractionBlock( 16, 32, 3, 32, 32, 3 ) ## (60-2-2)=56
        self.pool4_prior = nn.MaxPool2d(2) ## 56/2=28
        self.conv_encode5_prior = self.ExtractionBlock( 32, 64, 3, 64, 64, 3 ) ## (28-2-2)=24
        self.pool5_prior = nn.MaxPool2d(2) ## 24/2=12
        self.conv_encode6_prior = self.ExtractionBlock( 64, 128, 3, 128, 128, 3 ) ## （12-2-2）=8 
        self.pool6_prior = nn.MaxPool2d(2) ## 8/2=4
        self.conv_encode7_prior = self.ExtractionBlock( 128, 256, 3, 256, 256, 1 ) ## (4-2-0)=2
        self.pool7_prior = nn.MaxPool2d(2) ## 2/2=1
        self.conv_encode8_prior= self.ExtractionBlock( 256, 256, 1, 256, 256, 1 ) ## 1       

        
        ## Decode
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv1 = self.ExpansionBlock( 1024, 512, 1, 512, 512, 3 ) ## 4*4@512
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv2 = self.ExpansionBlock( 512, 256, 3, 256, 256, 3 ) ## 12*12@256
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv3 = self.ExpansionBlock( 256, 128, 3, 128, 128, 3 ) ## 40*40@48
        
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv4 = self.ExpansionBlock( 128, 64, 3, 64, 64, 3 ) ## 126@16
        
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv5 = self.ExpansionBlock( 64, 32, 3, 32, 32, 3 ) ## 512*512@1
        
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv6 = self.ExpansionBlock( 32, 16, 3, 16, 16, 3 ) ## 512*512@1
        
        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv7 = self.ExpansionBlock( 16, 16, 5, 16, 16, 5) ## 512*512@1
        
        self.finalconv  = self.FinalConv( 16, 16, 1, 16, 8, 3, 8, 1, 1 ) 
        
        
    def forward(self, image1,image2,image3, prior):
#        feature_conv = []
## Encode 
# image_seq_1
        DownConv1_img1 = self.conv_encode1(image1) #504@4
        temp = DownConv1_img1.squeeze(dim=2)
#        feature_conv.append(temp)
        DownConv1_img1 = self.conv_encode1_2(temp)
        DownConv1_pool_img1 = self.pool1(DownConv1_img1) #252@4
         
        DownConv2_img1 = self.conv_encode2(DownConv1_pool_img1) #248@8
        DownConv2_pool_img1 = self.pool2(DownConv2_img1) # 124@8
         
        DownConv3_img1 = self.conv_encode3( DownConv2_pool_img1 ) #120@16
        DownConv3_pool_img1 = self.pool3( DownConv3_img1 )   #60@16
           
        DownConv4_img1 = self.conv_encode4(DownConv3_pool_img1) #56@32
        DownConv4_pool_img1 = self.pool4(DownConv4_img1)#28@32
            
        DownConv5_img1 = self.conv_encode5(DownConv4_pool_img1) #24@64
        DownConv5_pool_img1 = self.pool5(DownConv5_img1)   #12@64
           
        DownConv6_img1 = self.conv_encode6(DownConv5_pool_img1) #8@128
        DownConv6_pool_img1=self.pool6(DownConv6_img1)   #4@128
         
        DownConv7_img1 = self.conv_encode7(DownConv6_pool_img1) #2@256
        DownConv7_pool_img1=self.pool7(DownConv7_img1)   #1@256
         
        DownConv8_img1 = self.conv_encode8(DownConv7_pool_img1) #1@256
        
        del DownConv1_pool_img1
        del DownConv2_pool_img1
        del DownConv3_pool_img1
        del DownConv4_pool_img1
        del DownConv5_pool_img1
        del DownConv6_pool_img1
        del DownConv7_pool_img1

# image_seq_2
        DownConv1_img2 = self.conv_encode1(image2) #504@4
        temp = DownConv1_img2.squeeze(dim=2)
#        feature_conv.append(temp)
        DownConv1_img2 = self.conv_encode1_2(temp)
        DownConv1_pool_img2 = self.pool1(DownConv1_img2) #252@4
         
        DownConv2_img2 = self.conv_encode2(DownConv1_pool_img2) #248@16
        DownConv2_pool_img2 = self.pool2(DownConv2_img2) # 124@16
 
        DownConv3_img2 = self.conv_encode3( DownConv2_pool_img2 ) #120@32
        DownConv3_pool_img2 = self.pool3( DownConv3_img2 )   #60@32
   
        DownConv4_img2 = self.conv_encode4(DownConv3_pool_img2) #56@64
        DownConv4_pool_img2 = self.pool4(DownConv4_img2)#28@64

        DownConv5_img2 = self.conv_encode5(DownConv4_pool_img2) #24@128
        DownConv5_pool_img2 = self.pool5(DownConv5_img2)   #12@128
   
        DownConv6_img2 = self.conv_encode6(DownConv5_pool_img2) #8@256
        DownConv6_pool_img2=self.pool6(DownConv6_img2)   #4@256
 
        DownConv7_img2 = self.conv_encode7(DownConv6_pool_img2) #2@512
        DownConv7_pool_img2=self.pool7(DownConv7_img2)   #1@512
 
        DownConv8_img2 = self.conv_encode8(DownConv7_pool_img2) #1@512
        
        del DownConv1_pool_img2
        del DownConv2_pool_img2
        del DownConv3_pool_img2
        del DownConv4_pool_img2
        del DownConv5_pool_img2
        del DownConv6_pool_img2
        del DownConv7_pool_img2

# image_seq_3
        DownConv1_img3 = self.conv_encode1(image3) #504@8
        temp = DownConv1_img3.squeeze(dim=2)
#        feature_conv.append(temp)
        DownConv1_img3 = self.conv_encode1_2(temp)
        DownConv1_pool_img3 = self.pool1(DownConv1_img3) #252@8
 
        DownConv2_img3 = self.conv_encode2(DownConv1_pool_img3) #248@16
        DownConv2_pool_img3 = self.pool2(DownConv2_img3) # 124@16
         
        DownConv3_img3 = self.conv_encode3( DownConv2_pool_img3 ) #120@32
        DownConv3_pool_img3 = self.pool3( DownConv3_img3 )   #60@32
           
        DownConv4_img3 = self.conv_encode4(DownConv3_pool_img3) #56@64
        DownConv4_pool_img3 = self.pool4(DownConv4_img3)#28@64
        
        DownConv5_img3 = self.conv_encode5(DownConv4_pool_img3) #24@128
        DownConv5_pool_img3 = self.pool5(DownConv5_img3)   #12@128
           
        DownConv6_img3 = self.conv_encode6(DownConv5_pool_img3) #8@256
        DownConv6_pool_img3 =self.pool6(DownConv6_img3)   #4@256
         
        DownConv7_img3 = self.conv_encode7(DownConv6_pool_img3) #2@512
        DownConv7_pool_img3 = self.pool7(DownConv7_img3)   #1@512
 
        DownConv8_img3 = self.conv_encode8(DownConv7_pool_img3) #1@512
        
        del DownConv1_pool_img3
        del DownConv2_pool_img3
        del DownConv3_pool_img3
        del DownConv4_pool_img3
        del DownConv5_pool_img3
        del DownConv6_pool_img3
        del DownConv7_pool_img3


## Encode- Prior
        
        Prior_DownConv1 = self.conv_encode1_prior(prior)
#        feature_conv.append(Prior_DownConv1)
        PriorDownConv1_pool1 = self.pool1(Prior_DownConv1)
         
        Prior_DownConv2 = self.conv_encode2_prior(PriorDownConv1_pool1)
        PriorDownConv2_pool2=self.pool2(Prior_DownConv2)
         
        Prior_DownConv3 = self.conv_encode3_prior(PriorDownConv2_pool2)   
        PriorDownConv3_pool3=self.pool3(Prior_DownConv3)  
         
        Prior_DownConv4 = self.conv_encode4_prior(PriorDownConv3_pool3) 
        PriorDownConv4_pool4 = self.pool4(Prior_DownConv4) 
           
        Prior_DownConv5 = self.conv_encode5_prior(PriorDownConv4_pool4)       
        PriorDownConv5_pool5 = self.pool5(Prior_DownConv5) 
         
        Prior_DownConv6 = self.conv_encode6_prior(PriorDownConv5_pool5)  
        PriorDownConv6_pool6 = self.pool6(Prior_DownConv6) 
         
        Prior_DownConv7 = self.conv_encode7_prior(PriorDownConv6_pool6)
        PriorDownConv7_pool7 = self.pool7(Prior_DownConv7)
        
        Prior_DownConv8 = self.conv_encode8_prior(PriorDownConv7_pool7)
        
        del PriorDownConv1_pool1
        del PriorDownConv2_pool2
        del PriorDownConv3_pool3
        del PriorDownConv4_pool4
        del PriorDownConv5_pool5
        del PriorDownConv6_pool6
        del PriorDownConv7_pool7
        
       
#Decode
        temp = torch.cat((Prior_DownConv8,DownConv8_img1, DownConv8_img2, DownConv8_img3), dim=1) #1@256*4
        up1 = self.up1(temp) # 2@256*4
        temp = torch.cat((Prior_DownConv7,DownConv7_img1,DownConv7_img2, DownConv7_img3), dim =1 ) #2@256*4
        Tconv_1 = self.Tconv1(up1+temp)  #4@512

 
        up2 = self.up2(Tconv_1) #8@512
        temp = torch.cat((Prior_DownConv6,DownConv6_img1,DownConv6_img2,DownConv6_img3), dim =1 )  #8@128*4
        Tconv_2 = self.Tconv2(up2+temp)  #12@256
         
        up3 = self.up3(Tconv_2) #24@256
        temp = torch.cat((Prior_DownConv5,DownConv5_img1,DownConv5_img2,DownConv5_img3), dim =1 ) #24@64*4
        Tconv_3 = self.Tconv3(up3+temp)  #28@128
        
        up4 = self.up4(Tconv_3) #56@128
        temp = torch.cat((Prior_DownConv4,DownConv4_img1,DownConv4_img2,DownConv4_img3), dim =1 ) #56@32*4
        Tconv_4 = self.Tconv4(up4+temp)  #60@64
         
        up5 = self.up5(Tconv_4) #120@64
        temp = torch.cat((Prior_DownConv3,DownConv3_img1,DownConv3_img2,DownConv3_img3), dim =1 ) #120@16*4
        Tconv_5 = self.Tconv5(up5+temp)  #124@32
         
        up6 = self.up6(Tconv_5) #248@32
        temp = torch.cat((Prior_DownConv2,DownConv2_img1,DownConv2_img2,DownConv2_img3), dim =1 )  #248@8*3
        Tconv_6 = self.Tconv6(up6+temp)  #252@16
         
        up7 = self.up7(Tconv_6) #504@16
        temp = torch.cat((Prior_DownConv1,DownConv1_img1,DownConv1_img2,DownConv1_img3), dim =1 ) # 504@4*4
        Tconv_7 = self.Tconv7(up7+temp)  #508@16
        
        
        out = self.finalconv(Tconv_7) # 508@16 --> 512@8 --> 512@1
         
        return out
    
## instantiation
	
#model = CycNnet()
#model = model.to(device)
#
#from TrainDataset_Circle_Nnet3D import *
#x_transform = T.ToTensor()
#y_transform = T.ToTensor()
#            
#  
#
#dataset_4DCBCT = TrainDataset_CircleNnet3D(
#        'E:/Pytorch/4DCBCT_TrainingData/'
#        ,'E:/Pytorch/4DCBCT_TrainingData/'
#        ,[101]
#        ,[66]
#        , transform = x_transform
#        , target_transform = y_transform
#        )
#
#dataloader = data.DataLoader(dataset_4DCBCT, batch_size=1, shuffle=True, num_workers=0)
#
#aa,bb,cc,dd,ee = next(iter(dataloader)) # aa,bb,cc均为[1,3,512,512] 
#
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#aa = aa.to(device)
#bb = bb.to(device)
#cc = cc.to(device)
#        
#dd = dd.to(device)
#ee = ee.to(device)
#
#prediction = model( aa, bb, cc, dd )




