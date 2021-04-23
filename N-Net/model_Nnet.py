## Nnet Architecture

import torch  ## the top-level pytorch package and tensor library
import torch.nn as nn
import torch.nn.functional as F


## Model Architecture
class Nnet(nn.Module):
    
    def ExtractionBlock( self, InChannel_1, OutChannel_1, Kernal_1, InChannel_2, OutChannel_2, Kernal_2 ):
        DownBlock = nn.Sequential(
                nn.Conv2d(InChannel_1,OutChannel_1,Kernal_1),
                nn.PReLU(),
                nn.Conv2d(InChannel_2, OutChannel_2, Kernal_2),
                nn.PReLU(),
                ) 
        return DownBlock
       
    
    def ExpansionBlock( self, InChannel1, OutChannel1, Kernel1, InChannel2, OutChannel2, Kernel2 ):
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
            ,nn.PReLU()
            ,nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2, padding=1 )
            ,nn.PReLU()
            ,nn.Conv2d( InChannel_3, OutChannel_3, Kernal_3, padding=0 )
            ,nn.PReLU()
            )
        return finalconv
   
    
    def __init__(self):
        super(Nnet, self).__init__()        
        
        ## Encode
        self.conv_encode1 = self.ExtractionBlock( 1, 8, 5, 8, 8, 5 ) ## (512-4-4)=504
        self.pool1 = nn.MaxPool2d(2) #504/2=252
        self.conv_encode2 = self.ExtractionBlock( 8, 16, 3, 16, 16, 3 ) ## (252-2-2)=248
        self.pool2 = nn.MaxPool2d(2) #248/2=124
        self.conv_encode3 = self.ExtractionBlock( 16, 32, 3, 32, 32, 3 ) ## (124-2-2)=120
        self.pool3 = nn.MaxPool2d(2) # 120/2=60
        self.conv_encode4 = self.ExtractionBlock( 32, 64, 3, 64, 64, 3 ) ## (60-2-2)=56
        self.pool4 = nn.MaxPool2d(2) ## 56/2=28
        self.conv_encode5 = self.ExtractionBlock( 64, 128, 3, 128, 128, 3 ) ## (28-2-2)=24
        self.pool5 = nn.MaxPool2d(2) ## 24/2=12
        self.conv_encode6 = self.ExtractionBlock( 128, 256, 3, 256, 256, 3 ) ## （12-2-2）=8 
        self.pool6 = nn.MaxPool2d(2) ## 8/2=4
        self.conv_encode7 = self.ExtractionBlock( 256, 512, 3, 512, 512, 1 ) ## (4-2-0)=2
        self.pool7 = nn.MaxPool2d(2) ## 2/2=1
        self.conv_encode8= self.ExtractionBlock( 512, 512, 1, 512, 512, 1 ) ## 1

        
        
        ## Decode
        self.up1 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.Tconv1 = self.ExpansionBlock( 1024, 512, 1, 512, 512, 3 ) ## 4*4@512
        
        self.up2 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.Tconv2 = self.ExpansionBlock( 512, 256, 3, 256, 256, 3 ) ## 12*12@256
        
        self.up3 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.Tconv3 = self.ExpansionBlock( 256, 128, 3, 128, 128, 3 ) ## 40*40@48
        
        self.up4 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.Tconv4 = self.ExpansionBlock( 128, 64, 3, 64, 64, 3 ) ## 126@16
        
        self.up5 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.Tconv5 = self.ExpansionBlock( 64, 32, 3, 32, 32, 3 ) ## 512*512@1
        
        self.up6 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.Tconv6 = self.ExpansionBlock( 32, 16, 3, 16, 16, 3 ) ## 512*512@1
        
        self.up7 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.Tconv7 = self.ExpansionBlock( 16, 16, 5, 16, 16, 5) ## 512*512@1
        

        self.finalconv  = self.FinalConv( 16, 16, 1, 16, 8, 3, 8, 1, 1 ) 
        
        
    def forward(self, image, prior):
#         per_out=[]
         ## Encode    
         DownConv1 = self.conv_encode1(image) #504@8
         DownConv1_pool1 = self.pool1(DownConv1) #252@8
         
         DownConv2 = self.conv_encode2(DownConv1_pool1) #248@16
         DownConv2_pool2=self.pool2(DownConv2) # 124@16
         
         DownConv3 = self.conv_encode3(DownConv2_pool2) #120@32
         DownConv3_pool3 = self.pool3(DownConv3)   #60@32
           
         DownConv4 = self.conv_encode4(DownConv3_pool3) #56@64
         DownConv4_pool4 = self.pool4(DownConv4)#28@64
            
         DownConv5 = self.conv_encode5(DownConv4_pool4) #24@128
         DownConv5_pool5 = self.pool5(DownConv5)   #12@128
           
         DownConv6 = self.conv_encode6(DownConv5_pool5) #8@256
         DownConv6_pool6=self.pool6(DownConv6)   #4@256
         
         DownConv7 = self.conv_encode7(DownConv6_pool6) #2@512
         DownConv7_pool7=self.pool7(DownConv7)   #1@512
         
         DownConv8 = self.conv_encode8(DownConv7_pool7) #1@512

         
         ## Encode- Prior
         Prior_DownConv1 = self.conv_encode1(prior)
         PriorDownConv1_pool1 = self.pool1(Prior_DownConv1)

         
         Prior_DownConv2 = self.conv_encode2(PriorDownConv1_pool1)
         PriorDownConv2_pool2=self.pool2(Prior_DownConv2)
         
         Prior_DownConv3 = self.conv_encode3(PriorDownConv2_pool2)   
         PriorDownConv3_pool3 = self.pool3(Prior_DownConv3)  
         
         Prior_DownConv4 = self.conv_encode4(PriorDownConv3_pool3) 
         PriorDownConv4_pool4 = self.pool4(Prior_DownConv4) 
           
         Prior_DownConv5 = self.conv_encode5(PriorDownConv4_pool4)       
         PriorDownConv5_pool5 = self.pool5(Prior_DownConv5) 
         
         Prior_DownConv6 = self.conv_encode6(PriorDownConv5_pool5)  
         PriorDownConv6_pool6 = self.pool6(Prior_DownConv6) 
         
         Prior_DownConv7 = self.conv_encode7(PriorDownConv6_pool6)
         PriorDownConv7_pool7 = self.pool7(Prior_DownConv7)

         Prior_DownConv8 = self.conv_encode8(PriorDownConv7_pool7)        

         # Decode
         temp = torch.cat((Prior_DownConv8,DownConv8), dim=1) #1@1024
         up1 = self.up1(temp) # 2@1024
         temp = torch.cat((Prior_DownConv7,DownConv7), dim =1 ) #2@1024
         Tconv_1 = self.Tconv1(up1+temp)  #4@512
        
         
         up2 = self.up2(Tconv_1) 
         temp = torch.cat((Prior_DownConv6,DownConv6), dim =1 ) 
         Tconv_2 = self.Tconv2(up2+temp)  #12@256
         
         up3 = self.up3(Tconv_2) 
         temp = torch.cat((Prior_DownConv5,DownConv5), dim =1 ) 
         Tconv_3 = self.Tconv3(up3+temp)  #28@128

         up4 = self.up4(Tconv_3) 
         temp = torch.cat((Prior_DownConv4,DownConv4), dim =1 ) 
         Tconv_4 = self.Tconv4(up4+temp)  #60@64
         
         up5 = self.up5(Tconv_4) 
         temp = torch.cat((Prior_DownConv3,DownConv3), dim =1 ) 
         Tconv_5 = self.Tconv5(up5+temp)  #124@32
         
         up6 = self.up6(Tconv_5) 
         temp = torch.cat((Prior_DownConv2,DownConv2), dim =1 ) 
         Tconv_6 = self.Tconv6(up6+temp)  #252@16
         
         up7 = self.up7(Tconv_6) 
         temp = torch.cat((Prior_DownConv1,DownConv1), dim =1 ) # 8@512
         Tconv_7 = self.Tconv7(up7+temp)  #512@16
            
    
         out = self.finalconv(Tconv_7) #512@1
         return out