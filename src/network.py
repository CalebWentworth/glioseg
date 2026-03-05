#define network
import torch
import snntorch as snn
import torch.nn as nn
from snntorch import utils, surrogate

class SpikingUnet(nn.Module):

    def __init__(self, in_channels ,out_chanels=2, beta=0.9):
        # layer parameters
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid()

        #encode l1

        self.enc1 = SNNConvBlock(in_channels, 16, beta, spike_grad)
        self.pool1 = nn.MaxPool3d(2)

        #encode l2
        self.enc2 = SNNConvBlock(16, 32, beta,spike_grad)
        self.pool2 = nn.MaxPool3d(2)

        #encode l3
        self.enc3 = SNNConvBlock(32, 64, beta, spike_grad)
        self.pool3 = nn.MaxPool3d(2)


        #bridge layer (bottom of the network)

        self.bridge = SNNConvBlock(64,128, beta, spike_grad)

        #decode l1
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = SNNConvBlock(128, 64, beta, spike_grad)

        #decode l2
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = SNNConvBlock(64, 32, beta, spike_grad)

        #decode l3
    
        self.up3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec3 = SNNConvBlock(32,16, beta, spike_grad)

        #output layer
        self.output_conv = nn.Conv3d(16,out_chanels, kernel_size=1)
        self.output_leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output = True)
    
    def forward(self,x):
        #this is the feed forward
        #encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        #bridge base layer
        b = self.bridge(p3)
        
        #decoder
        u1 = self.up1(b)
        cat1 = torch.cat([e3, u1], dim=1)
        d1 = self.dec1(cat1)
        
        u2 = self.up2(d1)
        cat2 = torch.cat([e2, u2], dim=1)
        d2 = self.dec2(cat2)
        
        u3 = self.up3(d2)
        cat3 = torch.cat([e1, u3], dim=1)
        d3 = self.dec3(cat3)
        
        spk, mem = self.output_leaky(self.output_conv(d3))
        return spk, mem
    
    def init_mem(self):
            utils.reset(self)


class SNNConvBlock(nn.Module):

    def __init__(self,in_channels,out_chanels, beta, spike_grad):
        
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_chanels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_chanels),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

            nn.Conv3d(out_chanels, out_chanels,kernel_size=3,padding=1),
            nn.BatchNorm3d(out_chanels),
            snn.Leaky(beta=beta,spike_grad=spike_grad, init_hidden=True)
        )

    def forward(self, x):
        return self.block(x)
        
