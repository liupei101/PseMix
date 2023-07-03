from typing import List
import torch
import torch.nn as nn
import math

from .deepmil import DeepMIL, DSMIL, TransMIL


##########################################
# Functions for loading models 
##########################################
def load_model(task:str, backbone:str, dims:List, **kws):
    if task == 'clf':
        if backbone == 'ABMIL':
            return Deep_ABMIL(dims, **kws)
        elif backbone == 'MaxMIL':
            return Deep_MaxMIL(dims, **kws)
        elif backbone == 'MeanMIL':
            return Deep_MeanMIL(dims, **kws)
        elif backbone == 'DSMIL':
            return Deep_DSMIL(dims, **kws)
        elif backbone == 'TransMIL':
            return Deep_TransMIL(dims, **kws)
        else:
            raise NotImplementedError("Backbone {} cannot be recognized".format(backbone))
    else:
        pass

def Deep_TransMIL(dims, **kws):
    #assert dims[0] == 1024 # input dim is 1024 in official TransMIL
    model = TransMIL(dims[0], dims[1], dims[2], **kws)

    return model

def Deep_DSMIL(dims, **kws):
    model = DSMIL(dims[0], dims[1], dims[2], **kws)

    return model
    
def Deep_ABMIL(dims, **kws):
    model = DeepMIL(dims[0], dims[1], dims[2], pooling='gated_attention', pred_head='default', **kws)

    return model

def Deep_MaxMIL(dims, **kws):
    model = DeepMIL(dims[0], dims[1], dims[2], pooling='max', pred_head='default', **kws)

    return model

def Deep_MeanMIL(dims, **kws):
    model = DeepMIL(dims[0], dims[1], dims[2], pooling='mean', pred_head='default', **kws)

    return model

##########################################
# Model weight initialization functions
##########################################
@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

@torch.no_grad()
def general_init_weight(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

def init_pytorch_defaults(m, version='041'):
    '''
    copied from AMDIM repo: https://github.com/Philip-Bachman/amdim-public/
    note from me: haven't checked systematically if this improves results
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False

