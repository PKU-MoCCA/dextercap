# models.py
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as tvm
from torchvision.models import resnet18, resnet34

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
        self.model = resnet18(num_classes=3)
        # self.model = resnet34(num_classes=3)

        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        return self.model(x)
    
class EdgeNet(nn.Module):
    def __init__(self):
        super(EdgeNet, self).__init__()
    
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        # self.model = resnet18(num_classes=3)
        self.model = resnet34(num_classes=hidden_dim)

        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=True
        )
                
        self.head = nn.Sequential(nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        y = self.model(x)        
        y = self.head(y)
        
        return y
    
class BlockNet(nn.Module):
    def __init__(self, label0_chars, label1_chars, blk_dirs):
        super(BlockNet, self).__init__()
    
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        self.model = resnet34(num_classes=hidden_dim*3)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=True
        )
        
        
        self.label_0_head = nn.Sequential(nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, label0_chars))
        self.label_1_head = nn.Sequential(nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, label1_chars))
        self.dir_head = nn.Sequential(nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, blk_dirs))

    def forward(self, x):
        y = self.model(x)
        
        y = y.reshape(-1, 3, self.hidden_dim)
        
        label_0 = self.label_0_head(y[:,0])
        label_1 = self.label_1_head(y[:,1])
        blk_dir = self.dir_head(y[:,2])
        
        return label_0, label_1, blk_dir

class UNet(nn.Module):
    def __init__(self, output_channel=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, output_channel, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        # Output
        out = self.out_conv(dec1)
        return out
    

class BlockNet_res50(nn.Module):
    def __init__(self, label0_chars, label1_chars, blk_dirs):
        super(BlockNet_res50, self).__init__()
        self.hidden_dim = 256
        self.feature_dim = 2048
        self.resnet50 = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential( *list(self.resnet50.children())[:-2] )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.label_0_head = nn.Sequential(nn.ELU(), nn.Linear(self.feature_dim, self.hidden_dim), nn.ELU(), nn.Linear(self.hidden_dim, label0_chars))
        self.label_1_head = nn.Sequential(nn.ELU(), nn.Linear(self.feature_dim, self.hidden_dim), nn.ELU(), nn.Linear(self.hidden_dim, label1_chars))
        self.dir_head = nn.Sequential(nn.ELU(), nn.Linear(self.feature_dim, self.hidden_dim), nn.ELU(), nn.Linear(self.hidden_dim, blk_dirs))



    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        label_0 = self.label_0_head(x)
        label_1 = self.label_1_head(x)
        blk_dir = self.dir_head(x)

        # print(label_0.shape, label_1.shape, blk_dir.shape)
        return label_0, label_1, blk_dir
