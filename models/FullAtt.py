import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .common import conv1x1_block, get_activation

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class MLP_module(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=None):
        super(MLP_module, self).__init__()
        self.mlp =nn.Sequential(
            Flatten(),
            nn.Linear(len(pool_types)*in_channels, len(pool_types)*in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(len(pool_types)*in_channels // reduction_ratio, in_channels)
            )
    def forward(self, x):
        return self.mlp(x)
class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=None,in_size=(224, 224)):
        super(ChannelGate, self).__init__()
        self.mlp = MLP_module(in_channels=in_channels,reduction_ratio=reduction_ratio,pool_types=pool_types)
        if in_size==(32, 32):
            self.mlp_32_32_xty = MLP_module(in_channels=32,reduction_ratio=2,pool_types=pool_types)
            self.mlp_16_16_xty = MLP_module(in_channels=16,reduction_ratio=2,pool_types=pool_types)
            self.mlp_8_8_xty = MLP_module(in_channels=8,reduction_ratio=2,pool_types=pool_types)
            self.mlp_4_4_xty = MLP_module(in_channels=4,reduction_ratio=2,pool_types=pool_types)
            
            self.mlp_32_32_ytx = MLP_module(in_channels=32,reduction_ratio=2,pool_types=pool_types)
            self.mlp_16_16_ytx = MLP_module(in_channels=16,reduction_ratio=2,pool_types=pool_types)
            self.mlp_8_8_ytx = MLP_module(in_channels=8,reduction_ratio=2,pool_types=pool_types)
            self.mlp_4_4_ytx = MLP_module(in_channels=4,reduction_ratio=2,pool_types=pool_types)
                        
        if in_size==(224, 224):    
            self.mlp_112_112_xty = MLP_module(in_channels=112,reduction_ratio=2,pool_types=pool_types)
            self.mlp_56_56_xty = MLP_module(in_channels=56,reduction_ratio=2,pool_types=pool_types)
            self.mlp_28_28_xty = MLP_module(in_channels=28,reduction_ratio=2,pool_types=pool_types)
            self.mlp_14_14_xty = MLP_module(in_channels=14,reduction_ratio=2,pool_types=pool_types)
            self.mlp_7_7_xty = MLP_module(in_channels=7,reduction_ratio=2,pool_types=pool_types)
            
            self.mlp_112_112_ytx = MLP_module(in_channels=112,reduction_ratio=2,pool_types=pool_types)
            self.mlp_56_56_ytx = MLP_module(in_channels=56,reduction_ratio=2,pool_types=pool_types)
            self.mlp_28_28_ytx = MLP_module(in_channels=28,reduction_ratio=2,pool_types=pool_types)
            self.mlp_14_14_ytx = MLP_module(in_channels=14,reduction_ratio=2,pool_types=pool_types)
            self.mlp_7_7_ytx = MLP_module(in_channels=7,reduction_ratio=2,pool_types=pool_types)
                                    
        self.in_size = in_size
        self.pool_types = pool_types
        if len(pool_types) > 1:
            self.pw_spatial = conv1x1_block(in_channels=len(pool_types), out_channels=1)        
            #self.BN = torch.nn.BatchNorm2d(num_features=out_channels)
    def forward(self, x):
        batch_size,num_channel,h,w = x.size()
        txy_out, xty_out, ytx_out = self.computing_AVG3D(x,self.pool_types,in_size=self.in_size)        
        xty_out = torch.transpose(xty_out,1,2)
        ytx_out = torch.transpose(ytx_out,1,3)
        x = F.relu(txy_out*(F.sigmoid(xty_out+ytx_out)))        
        return x
    def computing_AVG3D(self, x_pw, pool_type=None,in_size=None):
        channel_all_txy = None
        channel_all_xty = None
        channel_all_ytx = None
        spatial_all_txy = None                
        spatial_all_xty = None
        spatial_all_ytx = None
        batch_size,num_channel,h,w = x_pw.size()
        txy = x_pw;
        xty = torch.transpose(x_pw,1,2)
        ytx = torch.transpose(x_pw,1,3)
        for pool_type in self.pool_types:
            channel_pw_txy, spatial_pw_txy = self.get_statistic_feature(txy,pool_type)
            
            channel_pw_xty, spatial_pw_xty = self.get_statistic_feature(xty,pool_type)
            
            channel_pw_ytx, spatial_pw_ytx = self.get_statistic_feature(ytx,pool_type)
            
            if channel_all_txy is None:
                channel_all_txy = channel_pw_txy
            else:
                channel_all_txy = torch.cat((channel_all_txy,channel_pw_txy),1)
            if channel_all_xty is None:
                channel_all_xty = channel_pw_xty
            else:
                channel_all_xty = torch.cat((channel_all_xty,channel_pw_xty),1)
            if channel_all_ytx is None:
                channel_all_ytx = channel_pw_ytx
            else:
                channel_all_ytx = torch.cat((channel_all_ytx,channel_pw_ytx),1)
                
            if spatial_all_txy is None:
                spatial_all_txy = spatial_pw_txy
            else:
                spatial_all_txy = torch.cat((spatial_all_txy,spatial_pw_txy),1)         
                
            if spatial_all_xty is None:
                spatial_all_xty = spatial_pw_xty
            else:
                spatial_all_xty = torch.cat((spatial_all_xty,spatial_pw_xty),1) 
            if spatial_all_ytx is None:
                spatial_all_ytx = spatial_pw_ytx
            else:
                spatial_all_ytx = torch.cat((spatial_all_ytx,spatial_pw_ytx),1) 
        
        x_channel_txy = self.mlp(channel_all_txy)
        
        if in_size==(32, 32):
            if h == 32 and w == 32:
                x_channel_xty = self.mlp_32_32_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_32_32_ytx(channel_all_ytx)#channel weights
            if h == 16 and w == 16:
                x_channel_xty = self.mlp_16_16_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_16_16_ytx(channel_all_ytx)#channel weights
            if h == 8 and w == 8:
                x_channel_xty = self.mlp_8_8_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_8_8_ytx(channel_all_ytx)#channel weights
            if h == 4 and w == 4:
                x_channel_xty = self.mlp_4_4_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_4_4_ytx(channel_all_ytx)#channel weights
        if in_size==(224, 224):            
            if h == 112 and w == 112:
                x_channel_xty = self.mlp_112_112_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_112_112_ytx(channel_all_ytx)#channel weights
            if h == 56 and w == 56:
                x_channel_xty = self.mlp_56_56_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_56_56_ytx(channel_all_ytx)#channel weights
            if h == 28 and w == 28:
                x_channel_xty = self.mlp_28_28_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_28_28_ytx(channel_all_ytx)#channel weights
            if h == 14 and w == 14:
                x_channel_xty = self.mlp_14_14_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_14_14_ytx(channel_all_ytx)#channel weights
            if h == 7 and w == 7:                
                x_channel_xty = self.mlp_7_7_xty(channel_all_xty)#channel weights
                x_channel_ytx = self.mlp_7_7_ytx(channel_all_ytx)#channel weights
                        
        scale_txy = x_channel_txy.unsqueeze(2).unsqueeze(3).expand_as(txy)
        scale_xty = x_channel_xty.unsqueeze(2).unsqueeze(3).expand_as(xty)
        scale_ytx = x_channel_ytx.unsqueeze(2).unsqueeze(3).expand_as(ytx)
        if(len(self.pool_types) > 1): 
            x_spatial_txy = self.pw_spatial(spatial_all_txy)#spatial weights         
            x_spatial_xty = self.pw_spatial(spatial_all_xty)#spatial weights         
            x_spatial_ytx = self.pw_spatial(spatial_all_ytx)#spatial weights
            
           
            txy_out = (1 + F.sigmoid(scale_txy*x_spatial_txy)) * txy
            xty_out = (1 + F.sigmoid(scale_xty*x_spatial_xty)) * xty
            ytx_out = (1 + F.sigmoid(scale_ytx*x_spatial_ytx)) * ytx
        else:                        
            txy_out = (1 + F.sigmoid(scale_txy)) * txy
            xty_out = (1 + F.sigmoid(scale_xty)) * xty            
            ytx_out = (1 + F.sigmoid(scale_ytx)) * ytx            
        return txy_out, xty_out, ytx_out
    def get_statistic_feature(self, x, pool_type=None):
        squeeze_channel = None
        squeeze_spatial = None        
        if pool_type=='avg':
            squeeze_channel = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            squeeze_spatial = torch.mean(x,1).unsqueeze(1)
        elif pool_type=='max':
            squeeze_channel = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))                
            squeeze_spatial = torch.max(x,1)[0].unsqueeze(1)
        elif pool_type=='lp':
            squeeze_channel = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))                
            squeeze_spatial = torch.lp_pool2d(x,1)[0].unsqueeze(1)
        elif pool_type=='std':                
            stdf = torch.std(x,(2,3),unbiased=True)
            stdf = stdf.reshape(stdf.size()[0],stdf.size()[1],1,1)
            squeeze_channel = stdf
            squeeze_spatial = torch.std(x,1).unsqueeze(1)
        return squeeze_channel, squeeze_spatial
class AttTOP(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, no_spatial=False,in_size=None):
        super(AttTOP, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types,in_size=in_size)
        self.no_spatial=no_spatial        
    def forward(self, x):
        x_out = self.ChannelGate(x)        
        return x_out