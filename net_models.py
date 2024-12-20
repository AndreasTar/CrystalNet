import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet code is from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

# Part of code (positional encoding part) is from https://github.com/diolatzis/active-exploration

# Shared Components

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ThreeWayUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ThreeWayUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        
        
        self.up_group_1_0 = (Up(512, 256 // factor, bilinear))
        self.up_group_1_1 = (Up(256, 128 // factor, bilinear))
        self.up_group_1_2 = (Up(128, 64 // factor, bilinear))
        self.up_group_1_3 = (Up(64, 32, bilinear))
        self.up_group_1_4 = (OutConv(32, 3))

        self.up_group_2_0 = (Up(512, 256 // factor, bilinear))
        self.up_group_2_1 = (Up(256, 128 // factor, bilinear))
        self.up_group_2_2 = (Up(128, 64 // factor, bilinear))
        self.up_group_2_3 = (Up(64, 32, bilinear))
        self.up_group_2_4 = (OutConv(32, n_classes))

        self.up_group_3_0 = (Up(512, 256 // factor, bilinear))
        self.up_group_3_1 = (Up(256, 128 // factor, bilinear))
        self.up_group_3_2 = (Up(128, 64 // factor, bilinear))
        self.up_group_3_3 = (Up(64, 32, bilinear))
        self.up_group_3_4 = (OutConv(32, 2))                    


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        out1 = self.up_group_1_0(x5, x4)
        out1 = self.up_group_1_1(out1, x3)
        out1 = self.up_group_1_2(out1, x2)
        out1 = self.up_group_1_3(out1, x1)
        out1 = self.up_group_1_4(out1)

        out2 = self.up_group_2_0(x5, x4)
        out2 = self.up_group_2_1(out2, x3)
        out2 = self.up_group_2_2(out2, x2)
        out2 = self.up_group_2_3(out2, x1)
        out2 = self.up_group_2_4(out2)

        out3 = self.up_group_3_0(x5, x4)
        out3 = self.up_group_3_1(out3, x3)
        out3 = self.up_group_3_2(out3, x2)
        out3 = self.up_group_3_3(out3, x1)
        out3 = self.up_group_3_4(out3)

        # normal, oi, uv
        return out1, out2, out3
    
    
class RNet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(RNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=5,padding=2),
            nn.LeakyReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, out_channel, kernel_size=1)
        )
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class TNetBackBone(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(TNetBackBone, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
class TNet(nn.Module):
    def __init__(self,in_channel,hidden=32):
        super(TNet, self).__init__()
        self.tnet_f = TNetBackBone(in_channel,hidden)
        self.tnet_b = TNetBackBone(hidden,16)
        self.hidden = hidden
    def forward(self, g,rx):
        num_of_glass = g.shape[1]
        r_shape = (g.shape[0],self.hidden,g.shape[-2],g.shape[-1])
        result = torch.zeros(r_shape,device="cuda")
        for i in range(num_of_glass):
            result = result + self.tnet_f(torch.cat([rx, g[:,i,...]], 1) )
        result = self.tnet_b(result)
        return result

class CNet(nn.Module):
    def __init__(self,n_oi):
        super(CNet, self).__init__()
        self.inner_pos = fc_layer(in_features=3, out_features=16)
        self.unet = UNet(37,8)
        self.tnet = TNet(8 + 17)
        self.r_unet = ThreeWayUNet(24,n_oi)
        self.rnet_uv = RNet(2,2)
        self.rnet_normal = RNet(3,3)
    def forward(self, x,g):
        position = x[:,4:7, :, :]
        position = torch.moveaxis(position, 1, 3)
        
        pe = self.inner_pos(position)
        pe = torch.moveaxis(pe, 3, 1)
        x = torch.cat([x, pe], 1)
        rx = self.unet(x)
        tx = self.tnet(g,rx)
        rtx = torch.cat([rx, tx], 1)
        normal, oi, uv = self.r_unet(rtx)
        uv = self.rnet_uv(uv)
        normal = self.rnet_normal(normal)
        return normal, oi, uv
    

class CrystalNet(nn.Module):
    def __init__(self,n_oi):
        super(CrystalNet, self).__init__()
        self.cnet = CNet(n_oi)
        
    def forward(self, x,g):
        n,o,d = self.cnet(x,g)
        return n,o,d


# Helper Layers

def fc_layer(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU(inplace=True))
