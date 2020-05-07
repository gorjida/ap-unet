from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import pickle


class DataReader(Dataset):
    def __init__(self, rawDataPath, mode, scale=1E-8):

        self.mode = mode
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        with open(rawDataPath,"rb") as file:
            self.data = pickle.load(file)
        #self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
         #           if not file.startswith('.')]
        #logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return self.data['iiter']

    def __getitem__(self, i):
        associationMatrix = \
            self.data['Ipap'][i]['APschdul'][-1]['switch_mat']
        roundedAssociationMatrix = (np.around(associationMatrix, decimals=0))
        beta = self.data['lscale_beta'][i]/self.scale
        allocatedPower = self.data['pload'][i]['zzeta_opt']
        #allocatedPower is a vector, let reshape it to a 2D image
        (nAccess,nUsers) = np.shape(beta)
        temp = np.reshape(allocatedPower,[1,nUsers])
        reshapedAllocatedPower = np.ones([nAccess,1]).dot(temp)

        betaImage = np.reshape(beta,[1,nAccess,nUsers])
        powerImage = np.reshape(reshapedAllocatedPower,[1,nAccess,nUsers])

        if self.mode==0:
            outImage = np.concatenate([betaImage, powerImage], axis=0)
        elif self.mode==1:
            outImage = np.concatenate([betaImage, roundedAssociationMatrix], axis=0)
        else:
            outImage = betaImage


        return {'image': torch.from_numpy(outImage),
                'mask': torch.from_numpy(roundedAssociationMatrix)
                ,'power': torch.from_numpy(temp)}
