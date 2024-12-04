#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:35:06 2017

@author: raimondas
"""

from collections import OrderedDict
import os
from setuptools._distutils.dir_util import mkpath

import torch
import torch.nn as nn

from tensorboard import summary

# def checkpoint(model, step=None, epoch=None):
#     package = {
#         'epoch': epoch if epoch else 'N/A',
#         'step': step if step else 'N/A',
#         'state_dict': model.state_dict(),
#     }
#     return package

# def anneal_learning_rate(optimizer, lr):
#     optim_state = optimizer.state_dict()
#     optim_state['param_groups'][0]['lr'] = lr
#     optimizer.load_state_dict(optim_state)
#     print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

def load(model, fpath_model):
    print (fpath_model)
    if os.path.exists(fpath_model) :
        print ("Loading model: %s" % os.path.split(fpath_model)[-1])

        # Load the model package
        map_location = torch.device('cpu') if not torch.cuda.is_available() else torch.device("cuda")
        package = torch.load(fpath_model, map_location=map_location)

        # Retrieve the epoch information
        epoch = package.get('epoch', 0) + 1  # Default to 0 if 'epoch' is missing

        # Adjust variable names for loading on a CPU or single-GPU setup
        state_dict = package['state_dict']
        new_state_dict = {}

        for key in state_dict.keys():
            # Remove 'module.' prefix if it exists (from DataParallel models)
            new_key = key.replace('module.', '', 1) if key.startswith('module.') else key
            new_state_dict[new_key] = state_dict[key]

        # Load the adjusted state_dict into the model
        model.load_state_dict(new_state_dict)

        print ("done.")
    else:
        epoch = 1
        print ("Pretrained model not found")
    return model, epoch


# def calc_params(model):
#     all_params = OrderedDict()
#     params = model.state_dict()

#     for _p in params.keys():
#         #if not('ih_l0_reverse' in _p):
#         all_params[_p] = params[_p].nelement()
#     return all_params

# def param_summary(model, writer, step):
#     state = model.state_dict()
#     for _p in state.keys():
#         param = state[_p].cpu().numpy()

#         s = summary.histogram(_p, param.flatten())
#         writer.add_summary(s, global_step = step)

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=True, keep_prob=0.5):
        super(BatchRNN, self).__init__()
        self.batch_norm = batch_norm
        self.bidirectional = bidirectional

        rnn_bias = False if batch_norm else True
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=bidirectional,
                          batch_first=True,
                          bias=rnn_bias)
        self.batch_norm_op = SequenceWise(nn.BatchNorm1d(hidden_size))

        self.dropout_op = nn.Dropout(1-keep_prob)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.contiguous()
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
            x = x.contiguous()
        if self.batch_norm:
            x = self.batch_norm_op(x)
        x = self.dropout_op(x)
        return x


#%%
class gazeNET(nn.Module):
    def __init__(self, num_classes, seed=220617):
        super(gazeNET, self).__init__()
        model_architecture = {
            "conv_stack": [["conv1", 8, [    2, 11], [1, 1]], 
            ["conv2", 8, [    2, 11], [1, 1]]], 
            "rnn_stack": [[ "gru1", 64,  True, True], 
                          ["gru2", 64,True, True], 
                          ["gru3", 64, True, True]] }
        torch.manual_seed(seed)
        if (torch.cuda.device_count()>0):
            torch.cuda.manual_seed(seed)

        if 'conv_stack' in model_architecture.keys():
            ## convolutional stack
            conv_config = model_architecture['conv_stack']
            conv_stack = []
            #feat_dim = int(math.floor((config['sample_rate'] * 2*config['window_stride']) / 2) + 1)
            feat_dim = 2
            in_channels = 1
            for _conv in conv_config:
                name, out_channels, kernel_size, stride = _conv
                padding = map(lambda x: int(x/2), kernel_size)
                _conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=tuple(kernel_size), stride=tuple(stride),
                              padding = tuple(padding),
                              bias = False
                              )
                #init_vars.xavier_uniform(conv_op.weight, gain=np.sqrt(2))
                _conv = nn.Sequential(
                    _conv,
                    nn.BatchNorm2d(out_channels),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Dropout(1-0.75),
                )
                conv_stack.append((name, _conv))
                in_channels = out_channels
                feat_dim = feat_dim/stride[0]+1
            self.conv_stack = nn.Sequential(OrderedDict(conv_stack))
            rnn_input_size = int(feat_dim * out_channels)
        else:
            self.conv_stack = None
            rnn_input_size = 2

        ## RNN stack
        rnn_config = model_architecture['rnn_stack']
        rnn_stack = []
        for _rnn in rnn_config:
            name, hidden_size, batch_norm, bidirectional = _rnn
            _rnn = BatchRNN(input_size=rnn_input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, batch_norm=batch_norm,
                            keep_prob = 0.75)
            rnn_stack.append((name, _rnn))
            rnn_input_size = hidden_size
        self.rnn_stack = nn.Sequential(OrderedDict(rnn_stack))

        ## FC stack
        self.fc = nn.Sequential(
            SequenceWise(nn.Linear(hidden_size, num_classes, bias=False)),
        )
    ### forward
    def forward(self, x):
        if self.conv_stack is not None:
            x = self.conv_stack(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous()  # TxNxH

        x = self.rnn_stack(x)

        x = self.fc(x)
        return x