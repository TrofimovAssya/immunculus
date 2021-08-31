from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


class TCRDataset(Dataset):
    def __init__(self, root_dir='.', save_dir='.', data_file='filename'):
        # Get patient lists and split into train/test 
        child, cb = self.get_filelists()

        # Get tcr datasets using patient lists
        self.tcr_data = self.get_dataset_fromfile(child, cb, 'tcr_data_train.csv')

        # Encode the datasets into onehot tensors
        self.train_inputs = self.encode_inputs(self.tcr_data['cdr3aa'])
        self.targets = np.array(self.tcr_data['tcr_type'])

        # Splitting the dataset into train/valid/test
        permut = np.random.permutation(np.arange(self.tcr_data.shape[0]))
        cutoff1 = int(0.9*self.tcr_data.shape[0])

        train_ix = permut[:cutoff1]
        valid_ix = permut[cutoff1:]

        self.valid_inputs = self.train_inputs[valid_ix]
        self.valid_targets = self.targets[valid_ix]
        self.train_inputs = self.train_inputs[train_ix]
        self.targets = self.targets[train_ix]

    def get_filelists(self):
        pheno_britanova = pd.read_csv('../DATA/metadata.txt', sep='\t')
        child = pheno_britanova[np.logical_and(pheno_britanova['age'] <= 12,
                                               pheno_britanova['age'] > 0)]['file_name']
        cb = pheno_britanova[pheno_britanova['age'] == 0]
        cb = cb['file_name']
        child = list(child)[:-1]
        cb = list(cb)[:-1]
        return child, cb

    def get_dataset_fromfile(self, child, cb, filename):
        if filename not in os.listdir('../CDR3_lists/'):
            print('No cached dataset...')
            print('Making data...')
            cb_tcr = pd.DataFrame([])
            child_tcr = pd.DataFrame([])

            fdir = '../DATA/'
            for fname in tqdm(child):
                if child_tcr.empty:
                    child_tcr = pd.read_csv(f'{fdir}{fname}', sep='\t')
                    child_tcr = child_tcr[['cdr3aa', 'cdr3nt', 'v', 'j']]
                else:
                    a = pd.read_csv(f'{fdir}{fname}', sep='\t')
                    a = a[['cdr3aa', 'cdr3nt', 'v', 'j']]
                    child_tcr = pd.concat([child_tcr, a])

            for fname in tqdm(cb):
                if cb_tcr.empty:
                    cb_tcr = pd.read_csv(f'{fdir}{fname}', sep='\t')
                    cb_tcr = cb_tcr[['cdr3aa', 'cdr3nt', 'v', 'j']]
                else:
                    a = pd.read_csv(f'{fdir}{fname}', sep='\t')
                    a = a[['cdr3aa', 'cdr3nt', 'v', 'j']]
                    cb_tcr = pd.concat([cb_tcr, a])

            keep = np.array([len(i) >= 7 for i in pd.Series(list(child_tcr['cdr3aa']))])
            child_tcr = child_tcr[keep]

            keep = np.array([len(i) >= 7 for i in pd.Series(list(cb_tcr['cdr3aa']))])
            cb_tcr = cb_tcr[keep]

            keep = np.logical_not(['*' in i or '_' in i for i in pd.Series(list(cb_tcr['cdr3aa']))])
            cb_tcr = cb_tcr[keep]

            keep = np.logical_not(['*' in i or '_' in i for i in pd.Series(list(child_tcr['cdr3aa']))])
            child_tcr = child_tcr[keep]

            child_tcr = child_tcr[np.logical_not(child_tcr['cdr3aa'].isin(cb_tcr['cdr3aa']))]
            cb_tcr = cb_tcr[np.logical_not(cb_tcr['cdr3aa'].isin(child_tcr['cdr3aa']))]

            types = np.hstack((np.zeros(child_tcr.shape[0]), np.ones(cb_tcr.shape[0])))
            tcr_data = pd.concat([child_tcr, cb_tcr])
            tcr_data['tcr_type'] = types

            tcr_data.to_csv(f'../CDR3_lists/{filename}')
        else:
            print('Loading data')
            tcr_data = pd.read_csv(f'../CDR3_lists/{filename}', index_col=0)
        return tcr_data

    def encode_inputs(self, cdr3_list):
        nb_seq = len(cdr3_list)
        amino_acid = self.get_aminoacid()
        encoded_sequences = np.zeros((nb_seq, 27))

        for ix,seq in tqdm(enumerate(cdr3_list), total=nb_seq):
            for aix in range(min(len(seq),27)):
                residue = seq[aix]
                aa = amino_acid[residue]
                encoded_sequences[ix,aix] = aa

        return encoded_sequences


    def encode_as_onehot(self, cdr3_list):
        # the thing that will encode TCR as onehot tensors
        nb_seq = len(cdr3_list)
        amino_acid = self.get_aminoacid()
        encoded_sequences = np.zeros((nb_seq, 27, 20))
        for ix,seq in tqdm(enumerate(cdr3_list), total=nb_seq):
            for aix in range(min(len(seq),27)):
                residue = seq[aix]
                aa = amino_acid[residue]
                encoded_sequences[ix,aix,aa]+=1

        return encoded_sequences

    def toonehot_target(self, targets):
        encoded = np.zeros((len(targets),2))
        targets = np.array(targets)
        for ix in range(len(targets)):
            encoded[ix,int(targets[ix])] += 1
        return encoded

    def get_aminoacid(self):
        # aminoacids
        amino_acid = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N',
                     'P', 'Q','R', 'S', 'T', 'V', 'W', 'Y']
        amino_acid_dict = {}
        ix = 1
        for aa in amino_acid:
            amino_acid_dict[aa] = ix
            ix += 1
        return amino_acid_dict

    def __len__(self):
        # the thing that will be enumerated by the dataloader
        return self.train_inputs.shape[0]

    def __getitem__(self, idx):
        inputs = self.train_inputs[idx]
        targets = self.targets[idx]
        return [inputs, targets]

    def input_size(self):
        return self.train_inputs.shape[1]


def get_dataset(opt, exp_dir, test=False):

    # All of the different datasets.

    if opt.dataset == 'tcr':
        dataset = TCRDataset(root_dir=opt.data_dir, save_dir=exp_dir,
                             data_file=opt.data_file)
    else:
        raise NotImplementedError()

    dataloader = DataLoader(dataset, shuffle=False, batch_size=opt.batch_size, num_workers=1)

    return dataloader
