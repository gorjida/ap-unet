from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

def getDataSet(inputDir,labelDir,imgScale,valPercent):
    dataset = BasicDataset(inputDir, labelDir, imgScale)
    # We use va_percent for validation and the rest for training
    n_val = int(len(dataset) * valPercent)
    n_train = len(dataset) - n_val
    # We randomly split the data to train/validation
    train, val = random_split(dataset, [n_train, n_val])
    return(train,val)

def dataLoader(train,val,batchSize):
    trainLoader = DataLoader(train,
                              batch_size=batchSize,
                              shuffle=True, num_workers=1,
                              pin_memory=True)
    valLoader = DataLoader(val, batch_size=batchSize,
                            shuffle=False, num_workers=1,
                            pin_memory=True, drop_last=True)
    return (trainLoader,valLoader)