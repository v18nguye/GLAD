import torch
import pdb
import torch.nn as nn
from ._module import Encoder, Decoder
from abc import abstractmethod

class Base(nn.Module):
    '''
    @params
        cfg:
            model config
    '''
    def __init__(self, supp_d_x, supp_d_y, d_raw_x, scale, cfg):
        super(Base, self).__init__()
        self.d_raw_x = d_raw_x
        self.d_y = supp_d_y
        self.d_x_in = supp_d_x + d_raw_x
        self.scale = scale
        _ecf = cfg.enc
        _dcf = cfg.dec
        
        self.enc = Encoder(in_dim_X = self.d_x_in, 
                           in_dim_y = self.d_y,
                           in_dim_E = _ecf.in_dim_E,
                           hid_dim_X = _ecf.hid_dim_X,
                           hid_dim_E = _ecf.hid_dim_E,
                           hid_dim_y = _ecf.hid_dim_y,
                           n_layers = _ecf.n_layers,
                           n_head = _ecf.n_head)
        
        self.dec = Decoder( in_dim_X = _ecf.hid_dim_X,  
                            in_dim_y = _ecf.hid_dim_X,
                            out_dim_X = self.d_raw_x,
                            in_dim_E = _dcf.in_dim_E, 
                            hid_dim_X = _dcf.hid_dim_X,
                            hid_dim_E = _dcf.hid_dim_E,
                            hid_dim_y = _dcf.hid_dim_y,
                            n_layers = _dcf.n_layers,
                            n_head = _dcf.n_head,
                            scale = self.scale)
    
    @abstractmethod
    def forward(self):
        pass