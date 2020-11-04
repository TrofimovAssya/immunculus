#!/usr/bin/env python
import torch
import pdb
import numpy as np
import pandas as pd
from torch.autograd import Variable
import os
import argparse
import datasets
import models
import pickle
import time
import monitoring
#
def build_parser():
    parser = argparse.ArgumentParser(description="")

    ### Hyperparameter options
    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=260389, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=10000, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    ### Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--target-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['tcr'], help='Which dataset to use.')
    parser.add_argument('--batchsize', default=128,type=int, help='batch size')
    parser.add_argument('--tcr-size', default=27,type=int, help='tcr length')


    # Model specific options
    parser.add_argument('--tcr-conv-layers-sizes', default=[20,1,18], type=int, nargs='+', help='TCR-Conv net config.')
    parser.add_argument('--mlp-layers-size', default=[250, 75, 50, 25, 10], type=int, nargs='+', help='MLP config')
    parser.add_argument('--emb_size', default=10, type=int, help='The size of the embeddings.')
    parser.add_argument('--loss', choices=['PCC', 'MSE','RMSE'], default = 'MSE', help='The cost function to use')
    parser.add_argument('--weight-decay', default=0, type=float, help='Weight decay parameter.')
    parser.add_argument('--model', choices=['tcr'], help='Model to use')
    parser.add_argument('--cpu', action='store_true', help='True if no gpu to be used')
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="gpu selection")


    # Monitoring options
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')
    parser.add_argument('--reload', default=False, type=bool, help='should be true if reloading an experiment')

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)
    # TODO: set the seed
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    if exp_dir is None: # we create a new folder if we don't load.
        exp_dir = monitoring.create_experiment_folder(opt)

    # creating the dataset
    print ("Getting the dataset...")
    ### making sure that if the experiment is reloaded we are
    ### also reloading the shuffled list
    if opt.reload:
        dataset_shuffle = np.load(f'{exp_dir}/dataset_shufflelist.npy')
        valid_list = np.load(f'{exp_dir}/dataset_validlist.npy')
        batch_list = np.load(f'{exp_dir}/dataset_batchlist.npy')
    else:
        dataset_shuffle = None
        valid_list = None
        batch_list = None

    dataset = datasets.get_dataset(opt,exp_dir, opt.reload,dataset_shuffle,batch_list,valid_list)


    # Creating a model
    print ("Getting the model...")

    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt)

    def RMSE(x,y):
        return torch.sqrt(torch.mean((x-y)**2))

    def PCC(x,y):

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) *
                                     torch.sqrt(torch.sum(vy ** 2)))
        cost = 1-(cost**2)
        return cost

        #return torch.sqrt(torch.mean((yhat-y)**2))

    #Training optimizer and stuff
    if opt.loss=='MSE':
        criterion = torch.nn.MSELoss()
    elif opt.loss == 'PCC':
        criterion = PCC
    elif opt.loss == 'RMSE':
        criterion = RMSE

    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")


    loss_dict = {}
    loss_dict['train_losses'] = []
    loss_dict['valid_losses'] = []
    valid_list = dataset.dataset.valid_ixs
    datasetsize = (dataset.dataset.batchlist.shape[0])

    current_pt = opt.target_file.split('_')[0]

    original_dataset = pd.read_csv(f'{opt.data_dir}/tcrispublic_{current_pt}.tsv',
               index_col=0)
    shuffled_original = original_dataset.iloc[dataset.dataset.examplelist,:]
    shuffled_original.to_csv(f'{exp_dir}/dataset_shuffled.csv')

    np.save(f'{exp_dir}/dataset_shufflelist.npy', dataset.dataset.examplelist)

    np.save(f'{exp_dir}/dataset_batchlist.npy', dataset.dataset.batchlist)
    np.save(f'{exp_dir}/dataset_validlist.npy', valid_list)

    folderfiles = os.listdir(f'{exp_dir}')
    if not 'tcr_representation' in folderfiles:
        os.mkdir (f'{exp_dir}/tcr_representation')
    if not 'tcr_preditions' in folderfiles:
        os.mkdir (f'{exp_dir}/tcr_preditions')

    for t in range(epoch, opt.epoch):

        loss_dict = monitoring.update_loss_dict(loss_dict,start = True)

        for no_b, mini in enumerate(dataset):

            inputs, targets = mini[0], mini[1]

            inputs = Variable(inputs, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).float()

            if not opt.cpu:
                inputs = inputs.cuda(opt.gpu_selection)
                targets = targets.cuda(opt.gpu_selection)

            y_pred = my_model(inputs).float()

            targets = torch.reshape(targets,(targets.shape[1],1))

            loss = criterion(y_pred, targets)
            loss_save = loss.data.cpu().numpy().reshape(1,)[0]

            ### saving the tcr representation
            representation = my_model.encode_tcr(inputs.squeeze().permute(0,2,1))
            representation = representation.data.cpu().numpy()
            batchnb = dataset.dataset.batchlist[no_b]
            np.save(f'{exp_dir}/tcr_representation/{no_b}_{batchnb}_tcremb.npy',representation)

            ### saving the tcr predictions
            predictions = y_pred.data.cpu().numpy()
            real = targets.data.cpu().numpy()
            predictions = np.hstack((predictions,real))
            np.save(f'{exp_dir}/tcr_preditions/{no_b}_{batchnb}_tcrpred.npy',predictions)


            if no_b in valid_list:
                loss_dict['valid_losses_epoch'].append(loss_save)
                if no_b%100==0:
                    print (f'epoch {t}: {no_b}/{datasetsize}: Loss - {loss_save} (valid)')
            else:
                loss_dict['train_losses_epoch'].append(loss_save)
                if no_b%100==0:
                    print (f'epoch {t}: {no_b}/{datasetsize}: Loss - {loss_save}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        monitoring.update_loss_dict(loss_dict, start=False)
        np.save(f'{exp_dir}/validation_loss.npy',loss_dict['valid_losses'])
        np.save(f'{exp_dir}/training_loss.npy',loss_dict['train_losses'])


    print ('Done training!')
    print ('Doing testing...')
    ### The patient possibilities for testing
    possibilities = ['HIP19717', 'HIP03505']
    current_pt = opt.target_file.split('_')[0]
    test_pt = np.random.choice(possibilities)
    while current_pt == test_pt:
        test_pt = np.random.choice(possibilities)
    print (f'selected test patient is {test_pt}')
    suffix = opt.target_file.split('_')[1]
    opt.target_file =f'{test_pt}_{suffix}'

    suffix = '_'.join(opt.data_file.split('_')[1:])
    opt.data_file = f'{test_pt}_{suffix}'
    print (f'New datafiles: ')
    print (opt.data_file)
    print (opt.target_file)


    dataset = datasets.get_dataset(opt,exp_dir,reload=False)

    ###  saving the shuffled order as well as the shuffled dataset.
    original_dataset = pd.read_csv(f'{opt.data_dir}/tcrispublic_{test_pt}.tsv',
               index_col=0)
    shuffled_original = original_dataset.iloc[dataset.dataset.examplelist,:]
    shuffled_original.to_csv(f'{exp_dir}/test_dataset_shuffled_{test_pt}.csv')
    np.save(f'{exp_dir}/test_dataset_shufflelist_{test_pt}.npy', dataset.dataset.examplelist)
    np.save(f'{exp_dir}/test_dataset_batchlist_{test_pt}.npy', dataset.dataset.batchlist)

    if not 'test_tcr_representation' in folderfiles:
        os.mkdir (f'{exp_dir}/test_tcr_representation')
    if not 'test_tcr_preditions' in folderfiles:
        os.mkdir (f'{exp_dir}/test_tcr_preditions')



    for no_b, mini in enumerate(dataset):

        inputs, targets = mini[0], mini[1]

        inputs = Variable(inputs, requires_grad=False).float()
        targets = Variable(targets, requires_grad=False).float()

        if not opt.cpu:
            inputs = inputs.cuda(opt.gpu_selection)
            targets = targets.cuda(opt.gpu_selection)

        y_pred = my_model(inputs).float()

        targets = torch.reshape(targets,(targets.shape[1],1))

        loss = criterion(y_pred, targets)
        print(loss)

        ### saving the tcr representation
        representation = my_model.encode_tcr(inputs.squeeze().permute(0,2,1))
        representation = representation.data.cpu().numpy()
        batchnb = dataset.dataset.batchlist[no_b]
        np.save(f'{exp_dir}/test_tcr_representation/{no_b}_{batchnb}_tcremb_{test_pt}.npy',representation)

        ### saving the tcr predictions
        predictions = y_pred.data.cpu().numpy()
        np.save(f'{exp_dir}/test_tcr_preditions/{no_b}_{batchnb}_tcrpred_{test_pt}.npy',predictions)




if __name__ == '__main__':
    main()
