""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import math


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,n_output, mode,
                 inputImageDim, bilinear=True):
        super(UNet, self).__init__()
        xdim = inputImageDim[0]
        ydim = inputImageDim[1]
        if not math.log2(xdim).is_integer() or not math.log2(ydim).is_integer():
            print("Dimensions must be a power of 2")
            sys.exit(1)

        #mode (0) ==> AP_scheduling with power as the input
        #mode (1) ===> power allocation only (single channel)
        #mode (2) ====> joint power allocation and AP_scheduling
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_output = n_output
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64) #Base image-size: 100by20
        self.down1 = Down(64, 128)#Base image-size: 50by10
        xdim/=2
        ydim/=2
        self.down2 = Down(128, 256)#Base image-size: 25by5
        xdim/=2
        ydim/=2
        self.down3 = Down(256, 512)#Base image-size: 12by2
        xdim/=2
        ydim/=2
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)#Base image-size: 6by1
        xdim/=2
        ydim/=2
        if mode==1 or mode==2:
            #Flattening
            xdim = int(xdim)
            ydim = int(ydim)
            print("Dimension of final squeezed image=("+str(xdim)+","+str(ydim)+")")
            flattenedSize = (1024 // factor)*xdim*ydim
            self.flattened = Flatten()
            #Perceptron
            self.perceptron = Perceptron(flattenedSize,n_output,500)

        if mode==0 or mode==2:
            self.up1 = Up(1024, 512, bilinear)#Base image-size: 12by2
            self.up2 = Up(512, 256, bilinear)#Base image-size: 24by4
            self.up3 = Up(256, 128, bilinear)#Base image-size: 48by8
            self.up4 = Up(128, 64 * factor, bilinear)#Base image-size: 96by16
            self.outc = OutConv(64, n_classes)#Base image-size: 6by1

    def forward(self, x):

        predDict = {}
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #
        if self.mode==1 or self.mode==2:
            x = self.flattened(x5)
            powerOut = self.perceptron(x)
            predDict["power"] = powerOut


        if self.mode==0 or self.mode==2:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            predDict["AP"] = logits
        return predDict
