import torch.nn.functional as F
import pdb
from torch import nn


class CNNClassifier(nn.Module):

    def __init__(self, tcr_conv_layers_sizes=[20, 1, 18],
                 mlp_layers_size=[10], tcr_input_size=27,
                 nb_samples=1, emb_size=10, data_dir='.'):
        super(CNNClassifier, self).__init__()

        self.emb_size = emb_size
        self.tcr_input_size = tcr_input_size

        layers = []
        outsize = self.tcr_input_size

        for i in range(0, len(tcr_conv_layers_sizes), 3):
            layer = nn.Conv1d(in_channels=tcr_conv_layers_sizes[i+0],
                              out_channels=tcr_conv_layers_sizes[i+1],
                              kernel_size=tcr_conv_layers_sizes[i+2], stride=1)
            outsize = outsize - tcr_conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim1 = [(tcr_conv_layers_sizes[i+1]*(outsize))]

        self.tcr_conv_stack = nn.ModuleList(layers)
        dim1 = dim1[0]

        layers = []
        dim = [dim1] + mlp_layers_size
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(dim[-1], 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def encode_tcr(self, tcr):

        for layer in self.tcr_conv_stack:
            tcr = layer(tcr)

        return tcr

    def forward(self, x):

        # Get the embeddings
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        emb_tcr = self.encode_tcr(x)
        mlp_input = emb_tcr.reshape((emb_tcr.shape[0],
                                   emb_tcr.shape[1]*emb_tcr.shape[2]))
        #mlp_input = emb_tcr.squeeze()
        # mlp_input = emb_tcr.permute(1,0,2)

        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)
        #mlp_output = self.softmax(mlp_output)

        return mlp_output



class TransformerClassifier(nn.Module):

    def __init__(self,nb_tlayers=5, nb_theads=10,
                 mlp_layers_size=[10], tcr_input_size=27,
                 nb_samples=1, emb_size=10, data_dir='.'):
        super(TransformerClassifier, self).__init__()

        self.emb_size = emb_size
        self.tcr_input_size = tcr_input_size

        outsize = self.tcr_input_size
        # Embedding 20 amino acids. 1 embedding for empty
        self.embedding = nn.Embedding(21, self.emb_size)

        # Defining transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=nb_theads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=nb_tlayers)

        layers = []
        dim1 = self.emb_size * self.tcr_input_size
        dim = [dim1] + mlp_layers_size
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(dim[-1], 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def embed_tcr(self, tcr):

        # layer that will perform the embedding
        
        tcr = self.embedding(tcr)

        return tcr



    def forward(self, x):

        # Get the embeddings
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        emb_tcr = self.emb_tcr(x)
        emb_tcr =self.transformer_encoder(emb_tcr)
        mlp_input = emb_tcr.reshape((emb_tcr.shape[0],
                                   emb_tcr.shape[1]*emb_tcr.shape[2]))

        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)

        return mlp_output


def get_model(opt, model_state=None):

    if opt.model == 'cnn':
        model_class = CNNClassifier

        model = model_class(tcr_conv_layers_sizes=opt.tcr_conv_layers_sizes,
                            mlp_layers_size=opt.mlp_layers_size,
                            tcr_input_size=opt.tcr_size,
                            emb_size=opt.emb_size, data_dir=opt.data_dir)

    elif opt.model == 'transformer':
        model_class = TransformerClassifier

        model = model_class(nb_tlayers=opt.nb_transformer_layers,
                            nb_theads=opt.nb_transformer_heads,
                            mlp_layers_size=opt.mlp_layers_size,
                            tcr_input_size=opt.tcr_size,
                            emb_size=opt.emb_size, data_dir=opt.data_dir)

    else:
        raise NotImplementedError()

    if model_state is not None:
        model.load_state_dict(model_state)

    return model


