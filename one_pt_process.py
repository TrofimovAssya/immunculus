import numpy as np
import pandas as pd
import os
import sys
from scipy.special import softmax
files = os.listdir('.')
sample = 0
max_val = 0

max_val=27

print (f'The maximum length of CDR3 found is {max_val}')

amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
grantham = {'S' : [0, 110, 145, 74, 58, 99, 124, 56, 142, 155, 144, 112, 89,68, 46, 121, 65, 80, 135, 177],
            'R' : [110, 0, 102, 103, 71, 112, 96, 125, 97, 97, 77, 180, 29, 43,86, 26, 96, 54, 91, 101],
            'L' : [145, 102, 0, 98, 92, 96, 32, 138, 5, 22, 36, 198, 99, 113,153, 107, 172, 138, 15, 61],
            'P' : [74, 103, 98, 0, 38, 27, 68, 42, 95, 114, 110, 169, 77, 76,91, 103, 108, 93, 87, 147],
            'T' : [58, 71, 92, 38, 0, 58, 69, 59, 89, 103, 92, 149, 47, 42, 65,78, 85, 65, 81, 128],
            'A' : [99, 112, 96, 27, 58, 0, 64, 60, 94, 113, 112, 195, 86, 91,111, 106, 126, 107, 84, 148],
            'V' : [124, 96, 32, 68, 69, 64, 0, 109, 29, 50, 55, 192, 84, 96,133, 97, 152, 121, 21, 88],
            'G' : [56, 125, 138, 42, 59, 60, 109, 0, 135, 153, 147, 159, 98,87, 80, 127, 94, 98, 127, 184],
            'I' : [142, 97, 5, 95, 89, 94, 29, 135, 0, 21, 33, 198, 94, 109,149, 102, 168, 134, 10, 61],
            'F' : [155, 97, 22, 114, 103, 113, 50, 153, 21, 0, 22, 205, 100,116, 158, 102, 177, 140, 28, 40],
            'Y' : [144, 77, 36, 110, 92, 112, 55, 147, 33, 22, 0, 194, 83, 99,143, 85, 160, 122, 36, 37],
            'C' : [112, 180, 198, 169, 149, 195, 192, 159, 198, 205, 194, 0,174, 154, 139, 202, 154, 170, 196, 215],
            'H' : [89, 29, 99, 77, 47, 86, 84, 98, 94, 100, 83, 174, 0, 24, 68,32, 81, 40, 87, 115],
            'Q' : [68, 43, 113, 76, 42, 91, 96, 87, 109, 116, 99, 154, 24, 0,46, 53, 61, 29, 101, 130],
            'N' : [46, 86, 153, 91, 65, 111, 133, 80, 149, 158, 143, 139, 68,46, 0, 94, 23, 42, 142, 174],
            'K' : [121, 26, 107, 103, 78, 106, 97, 127, 102, 102, 85, 202, 32,53, 94, 0, 101, 56, 95, 110],
            'D' : [65, 96, 172, 108, 85, 126, 152, 94, 168, 177, 160, 154, 81,61, 23, 101, 0, 45, 160, 181],
            'E' : [80, 54, 138, 93, 65, 107, 121, 98, 134, 140, 122, 170, 40,29, 42, 56, 45, 0, 126, 152],
            'M' : [135, 91, 15, 87, 81, 84, 21, 127, 10, 28, 36, 196, 87, 101,142, 95, 160, 126, 0, 67],
            'W' : [177, 101, 61, 147, 128, 148, 88, 184, 61, 40, 37, 215, 115,130, 174, 110, 181, 152, 67, 0]}

def get_onehot(seq, max_length, amino_acids):
    out_vector = np.zeros((max_length, len(amino_acids)))
    seq = [i for i in seq]
    for ix,i in enumerate(seq):
        if i in amino_acids:
            out_vector[ix,amino_acids.index(i)]+=1
    return out_vector

def get_grantham(seq, max_length, grantham):
    out_vector = np.zeros((max_length, len(grantham)))
    seq = [i for i in seq]
    for ix,i in enumerate(seq):
        out_vector[ix,:]+=grantham[i]
    return out_vector

def get_grantham_zeros(seq, max_length, grantham):
    out_vector = np.zeros((max_length, len(grantham)))
    seq = [i for i in seq]
    for ix,i in enumerate(seq):
        out_vector[ix,:]+=(1-(grantham[i]/np.max(grantham[i])))
    return out_vector

def cdr3_to_onehot(seq_table, max_length, amino_acids):
    out_data = np.zeros((len(seq_table), max_length, len(amino_acids)))
    for ix, seq in enumerate(seq_table):
        out_data[ix,:,:] = get_onehot(seq, max_length, amino_acids)
    return out_data

def cdr3_to_grantham_zeros(seq_table, max_length, grantham):
    out_data = np.zeros((len(seq_table), max_length, len(grantham.keys())))
    for ix, seq in enumerate(seq_table):
        if not '*' in seq:
            out_data[ix,:,:] = get_grantham_zeros(seq, max_length, grantham)
    return out_data


def cdr3_to_grantham(seq_table, max_length, grantham):
    out_data = np.zeros((len(seq_table), max_length, len(grantham.keys())))
    for ix, seq in enumerate(seq_table):
        if not '*' in seq:
            out_data[ix,:,:] = get_grantham(seq, max_length, grantham)
    return out_data

save_directory = 'dataset'
fname = sys.argv[1]
sample = sys.argv[2]
print ('Loading file...')
a = pd.read_csv(f'{save_directory}/{fname}',index_col=0)
aa = list(a.index)
targets = a['0']
print ('Converting to onehot...')
onehot_tcr = cdr3_to_onehot(list(aa), max_val, amino_acids)
print ('Converting to grantham...')
grantham_tcr = cdr3_to_grantham(list(aa), max_val, grantham)
print ('Converting to grantham with zeros...')
grantham_tcr_sm = cdr3_to_grantham_zeros(list(aa), max_val, grantham)

print ('Saving to file...')
np.save(f'{save_directory}/{sample}_tcr_oh.npy', onehot_tcr)
np.save(f'{save_directory}/{sample}_tcr_gd.npy', grantham_tcr)
np.save(f'{save_directory}/{sample}_tcr_gd_sm.npy', grantham_tcr_sm)
np.save(f'{save_directory}/{sample}_targets.npy', targets)
