import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import random
import torch
import shutil
import models
import datetime
import pandas as pd
import matplotlib.pyplot as plt


def save_loss(loss_dict,exp_dir):
    np.save(f'{exp_dir}/training_loss.npy',loss_dict['train_losses'])
    if 'valid_losses' in loss_dict:
        np.save(f'{exp_dir}/validation_loss.npy',loss_dict['valid_losses'])


def create_experiment_folder(opt):

    params = vars(opt).copy()
    params = str(params)

    # create a experiment folder
    this_hash = random.getrandbits(128)
    this_hash = "%032x" % this_hash # in hex

    exp_dir = os.path.join(opt.save_dir, this_hash)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    f = open(os.path.join(exp_dir,'run_parameters'), 'w')
    f.write(params+'\n')
    f.close()
    print (vars(opt))
    print (f"Saving the everything in {exp_dir}")

    with open(os.path.join(opt.save_dir, 'experiment_table.txt'), 'a') as f:
        f.write('time: {} folder: {} experiment: {}\n'.format(datetime.datetime.now(), this_hash, params))

    return exp_dir


def save_checkpoint(model, optimizer, epoch, opt, exp_dir, filename='checkpoint.pth.tar'):

    state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'opt' : opt
        }

    filename = os.path.join(exp_dir, filename)
    torch.save(state, filename)

def load_checkpoint(load_folder, opt, filename='checkpoint.pth.tar'):

    # Model
    model_state = None

    # Epoch
    epoch = 0

    # Optimizser
    optimizer_state = None

    # Options
    new_opt = opt

    # Load the states if we saved them.
    if opt.load_folder:

        # Loading all the state
        filename = os.path.join(load_folder, filename)
        if os.path.isfile(filename):
            print (f"=> loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']

            # Loading the options
            new_opt = checkpoint['opt']
            new_opt.gpu_selection=opt.gpu_selection
            print(f"Loading the model with these parameters: {new_opt}")

            # Loading the state
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            epoch = checkpoint['epoch']

            # We override some of the options between the runs, otherwise it might be a pain.
            new_opt.epoch = opt.epoch

            print(f"=> loaded checkpoint '{filename}' (epoch {epoch})")
        else:
            print(f"=> no checkpoint found at '{filename}'")

    # Get the network
    my_model = models.get_model(new_opt, model_state)

    ### Moving the model to GPU if it was on the GPU according to the opts.
    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # Get the optimizer
    #optimizer = torch.optim.RMSprop(my_model.parameters(), lr=new_opt.lr, weight_decay=new_opt.weight_decay)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=new_opt.lr, weight_decay=new_opt.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    print ("Our model:")
    print (my_model)

    return my_model, optimizer, epoch, new_opt
