import torch.nn as nn
import torch
from unet.unet_parts import *
from unet.unet_model_AP_Scheduling import *

if __name__=="__main__":

    crit1 = nn.BCEWithLogitsLoss()
    crit2 = nn.MSELoss()

    input = torch.rand(2,2,128,32)
    truth1 = torch.rand(2,1,128,32)
    truth2 = torch.rand(2,32)
    unetModel = UNet(2,1,32,2,[128,32])
    predDict = unetModel(input)
    loss1 = crit1(truth1,predDict["AP"])
    loss2 = crit2(truth2,predDict["power"])
    print(loss1+loss2)
