
import pickle
import numpy as np
import scipy.misc
from utils.data_reader import DataReader

#def readRecords(path):

if __name__=="__main__":
    mainFolder = "/home/ali/SharedFolder/detector_test/unetOptimization" \
                 "/measurement_campaign_20200430/data/"
    imageFolder = mainFolder + "imgs/"
    powerFolder = mainFolder + "powers/"
    assocFolder = mainFolder + "assoc/"
    path = mainFolder + \
           "res_UL_HP10m-K16-M128-sh0_Opt(IEQpower-lmda0.0,maxMinSNR,UL-bisec-Power(lp)-IPAP(iib)-isRoun0,sci-int,sci-int,1229-76-0,InitAssMat-sta-205).pkl"
    dataSet = DataReader(path)
    sys.exit(1)


    with open(path,"rb") as file:
        data = pickle.load(file)
    numIterations = data['iiter']
    for i in np.arange(0,numIterations,1):
        imageFile = "sample_"+str(i)
        associationMatrix = \
            data['Ipap'][i]['APschdul'][-1]['switch_mat']
        roundedAssociationMatrix = (np.around(associationMatrix, decimals=0))
        beta = np.log10(data['lscale_beta'][i])
        allocatedPower = data['pload'][i]['zzeta_opt']

        scipy.misc.toimage(beta).save(imageFile+".jpg")
        scipy.misc.toimage(associationMatrix).save(imageFile+"_mask.jpg")
        scipy.misc.toimage(allocatedPower).save(imageFile+"_power.jpg")
        print(i)





