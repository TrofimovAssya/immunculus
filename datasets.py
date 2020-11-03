from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import pdb

class TCRDataset(Dataset):
    """TCR dataset"""

    def __init__(self,root_dir='.',save_dir='.',data_file='data.npy',
                 target_file = 'target.npy', batchsize=128, test=False):


        data_path = os.path.join(root_dir, data_file)
        target_path = os.path.join(root_dir, target_file)
        self.batchsize = batchsize
        # Load the dataset
        self.data = np.load(data_path)
        self.target = np.load(target_path)
        ### dividing the target by the max target
        self.target /=np.max(self.target)
        self.target *=10
        ### permuting the dataset 
        examplelist = np.arange(self.data.shape[0])
        self.examplelist = np.random.permutation(examplelist)
        ### calculating the number of batches
        ### ceiling division (upside down floor)
        nb_batches = -(-self.examplelist.shape[0]//self.batchsize)
        batchlist = np.arange(nb_batches)
        self.batchlist = np.random.permutation(batchlist)

        ### taking 80% cutoff from the batches
        cutoff = int(0.8*self.batchlist.shape[0])
        self.valid_ixs = self.batchlist[cutoff:]


    def __len__(self):
        ### the thing that will be enumerated by the dataloader
        return len(self.batchlist)

    def __getitem__(self, idx):

        idx_s = self.batchsize*idx
        idx_e = self.batchsize*(idx+1)


        ### getting the data indices from the batch
        idx_keep = self.examplelist[idx_s:idx_e]

        ### getting the data from the selected data indices
        sample = self.data[idx_keep,:,:]
        label = self.target[idx_keep]

        sample = [sample, label]

        return sample

    def input_size(self):
        return self.examplelist.shape[0]


def get_dataset(opt,exp_dir, test=False):

    # All of the different datasets.

    if opt.dataset == 'tcr':
        dataset = TCRDataset(root_dir=opt.data_dir, save_dir = exp_dir,
                             data_file = opt.data_file, target_file =
                             opt.target_file, batchsize = opt.batchsize, 
                            test=test)
    else:
        raise NotImplementedError()

    dataloader = DataLoader(dataset, shuffle=False,num_workers=1)

    return dataloader


    return dataloader
