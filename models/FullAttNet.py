import torch.nn
import torch.nn.init 

from .common import conv3x3_dw_blockAll, conv1x1_block, conv3x3_block, conv3x3_dw_block, Classifier
from .FullAtt import *

class DwS_block(torch.nn.Module):    
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,pool_types,in_size=None):
        super().__init__()        
        self.pw1 = conv1x1_block(in_channels=in_channels,
                        out_channels=in_channels,                                
                        use_bn=False,
                        activation=None)        
        self.dw1 = conv3x3_dw_blockAll(channels=in_channels, stride=stride)
        if stride == 1:
            self.dw2 = conv3x3_dw_blockAll(channels=in_channels, stride=stride + 1)#, padding=57)
            self.dw3 = conv3x3_dw_blockAll(channels=in_channels, stride=stride + 2)#, padding=112)
        
        self.pw2 = conv1x1_block(in_channels=in_channels, out_channels=out_channels)
        if stride == 2 or in_channels != out_channels:
            self.pointwise_residual = conv1x1_block(in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride)
        self.AttTOP = AttTOP(out_channels, 16, pool_types=pool_types,in_size=in_size)        
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
    def forward(self, x):
        residual = x
        x = self.pw1(x)        
        x1 = self.dw1(x)        
        batch_size,num_channel,h,w = x1.size()
        if self.stride == 1:         
            x2 = self.dw2(x)        
            x2 = F.interpolate(x2,size=h,mode='nearest')
            x3 = self.dw3(x)        
            x3 = F.interpolate(x3,size=h,mode='nearest')
            x = F.relu(x1 * F.sigmoid(x2 * x3))
        else:
            x = x1
        x = self.pw2(x)
        x = self.AttTOP(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:            
            residual = self.pointwise_residual(residual)        
            x = x + residual
        return x

class AttTOPNet(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True,pool_types=None):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module("unit{}".format(unit_id + 1), DwS_block(in_channels=in_channels, out_channels=unit_channels, stride=stride,pool_types=pool_types,in_size=in_size))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        
        self.final_conv_channels = 1024#Thanh Tuan Add end
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))        
        self.backbone.add_module("dropout1",torch.nn.Dropout2d(0.2))#with dropout
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.backbone.add_module("dropout2",torch.nn.Dropout2d(0.2))#with dropout
        in_channels = self.final_conv_channels

        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():            
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)            
            elif isinstance(module, torch.nn.Linear):                
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):                
                module.weight.data.fill_(1)
                module.bias.data.zero_()            
                    
        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
def build_AttTOPNet(num_classes, width_multiplier=1.0, cifar=False,pool_types=None):

    init_conv_channels = 32    
    channels = [[64], [64, 128,128], [256,256,256,256], [256, 256], [512]]

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    if width_multiplier != 1.0:
        channels = [[int(unit * width_multiplier) for unit in stage] for stage in channels]
        init_conv_channels = int(init_conv_channels * width_multiplier)

    return AttTOPNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size,pool_types=pool_types)
