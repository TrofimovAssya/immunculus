import argparse
import datasets
import monitoring
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm


def build_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser(description="")

    # Hyperparameter options
    parser.add_argument('--epoch', default=1000, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=260389, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=100, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    # Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--target-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', default='tcr', choices=['tcr'], help='Which dataset to use.')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--tcr-size', default=27, type=int, help='tcr length')

    # Model specific options
    parser.add_argument('--tcr-conv-layers-sizes', default=[20, 20, 10, 20, 1,
                                                           9], type=int, nargs='+', help='TCR-Conv net config')
    parser.add_argument('--mlp-layers-size', default=[150, 50, 10], type=int, nargs='+', help='MLP config')
    parser.add_argument('--nb-transformer-layers', default=5, type=int, help='nb transformer layers')
    parser.add_argument('--nb-transformer-heads', default=10, type=int, help='nb transformer heads')
    parser.add_argument('--emb_size', default=10, type=int, help='The size of the embeddings.')
    parser.add_argument('--loss', choices=['NLL', 'MSE'], default='MSE', help='The cost function to use')
    parser.add_argument('--weight-decay', default=0, type=float, help='Weight decay parameter.')
    parser.add_argument('--model', default='cnn', choices=['cnn','transformer'], help='Model to use')
    parser.add_argument('--cpu', action='store_true', help='True if no gpu to be used')
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="gpu selection")

    # Monitoring options
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')

    return parser


def parse_args(argv):
    """creating the opt dictionary."""
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt


def main(argv=None):
    """Main."""
    opt = parse_args(argv)
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    if exp_dir is None:  # we create a new folder if we don't load.
        exp_dir = monitoring.create_experiment_folder(opt)

    # creating the dataset
    print("Getting the dataset...")
    dataset = datasets.get_dataset(opt, exp_dir)

    # Creating a model
    print("Getting the model...")

    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt)

    # Training optimizer and stuff
    criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()


    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")

    monitoring_dic = {}
    monitoring_dic['train_loss'] = []
    monitoring_dic['valid_loss'] = []
    monitoring_dic['train_accuracy'] = []
    monitoring_dic['valid_accuracy'] = []
    max_accuracy = 0
    min_loss = 10000
    patience = 5

    for t in range(epoch, opt.epoch):
        if patience==0:
            break
        thisepoch_trainloss = []
        thisepoch_trainaccuracy = []

        with tqdm(dataset, unit="batch") as tepoch:
            for mini in tepoch:

                tepoch.set_description(f"Epoch {t}")
                inputs, targets = mini[0], mini[1]

                inputs = Variable(inputs, requires_grad=False).long()
                targets = Variable(targets, requires_grad=False).long()

                if not opt.cpu:
                    inputs = inputs.cuda(opt.gpu_selection)
                    targets = targets.cuda(opt.gpu_selection)

                optimizer.zero_grad()
                y_pred = my_model(inputs).float()
                #y_pred = torch.reshape(y_pred, (y_pred.shape[0], ))

                #targets = torch.reshape(targets, (targets.shape[1], 1))

                loss = criterion(y_pred, targets)
                to_list = loss.cpu().data.numpy().reshape((1, ))[0]
                thisepoch_trainloss.append(to_list)


                loss.backward()
                optimizer.step()
                accuracy = np.sum(np.argmax(y_pred.cpu().data.numpy(),axis=1) ==
                       targets.cpu().data.numpy())/targets.shape[0]
                thisepoch_trainaccuracy.append(accuracy)
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

        inputs = dataset.dataset.valid_inputs
        inputs = torch.FloatTensor(inputs)
        inputs = Variable(inputs, requires_grad=False).long()


        targets = dataset.dataset.valid_targets
        targets = torch.FloatTensor(targets)
        targets = Variable(targets, requires_grad=False).long()

        if not opt.cpu:
            inputs = inputs.cuda(opt.gpu_selection)
            targets = targets.cuda(opt.gpu_selection)

        with torch.no_grad():
            y_pred = my_model(inputs)
            vloss = criterion(y_pred, targets)

        accuracy = np.sum(np.argmax(y_pred.cpu().data.numpy(),axis=1) ==
                          targets.cpu().data.numpy())/targets.shape[0]
        split = np.sum(targets.cpu().data.numpy())/targets.shape[0]

        vloss = vloss.cpu().data.numpy().reshape((1,))[0]
        if vloss<min_loss:
            monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
            print(f'*** new min loss*** Validation loss: epoch {t}, loss: {vloss}, accuracy: {accuracy}, split {split}')
            min_loss = vloss
            patience = 5
        else:

            print(f'Validation loss: epoch {t}, loss: {vloss}, accuracy:{accuracy}, split {split}')
            patience -= 1

        monitoring_dic['valid_loss'].append(vloss)
        monitoring_dic['valid_accuracy'].append(accuracy)
        monitoring_dic['train_loss'].append(np.mean(thisepoch_trainloss))
        monitoring_dic['train_accuracy'].append(np.mean(thisepoch_trainaccuracy))
        trainacc=np.mean(thisepoch_trainaccuracy)
        print (f'Training accuracy: {trainacc}')
        np.save(f'{exp_dir}/train_loss.npy',monitoring_dic['train_loss'])
        np.save(f'{exp_dir}/valid_loss.npy',monitoring_dic['valid_loss'])
        np.save(f'{exp_dir}/train_accuracy.npy',monitoring_dic['train_accuracy'])
        np.save(f'{exp_dir}/valid_accuracy.npy',monitoring_dic['valid_accuracy'])
    print ('Done training!')

if __name__ == '__main__':
    main()
