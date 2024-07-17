import pdb
import torch
import torch.nn as nn
import math
from ._layer import Mlp, XEyTransformerLayer, Mlp_old
from ..data._add_spec_feat import add_supp_feat
from ._lib import mask_x, mask_adjs, pow_tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_t):
        super().__init__()
        self.d_t = d_t
        
    def forward(self, t_position):
        """
        Arguments:
            t_position: Tensor int, shape ``[batch, 1]``
        """
        _b = t_position.shape[0]
        _device = t_position.device
        
        div_term = torch.exp(torch.arange(0, self.d_t, 2, device=_device) * (-math.log(10000.0) / self.d_t))
        pe = torch.zeros(_b, 1, self.d_t, device=_device) 
        pe[:, 0, 0::2] = torch.sin(t_position * div_term)
        pe[:, 0, 1::2] = torch.cos(t_position * div_term)
        
        return pe
     
class FSQuantizer(nn.Module):
    '''Finite Scalar Quantizer
    @params
        q_level:
            quan level
        d_zx: 
            node-embedding size
    '''
    def __init__(self, q_level, d_zx, max_node_num, act, depth, is_norm=True):
        super(FSQuantizer, self).__init__()
        self.q_level = q_level
        self.enc2q = Mlp_old(d_zx, len(q_level), len(q_level), max_node_num, depth=depth, act=act)
        self.q2dec = Mlp_old(len(q_level), d_zx, d_zx, max_node_num, depth=depth, act=act)
        
    def forward(self, z0, mask):
        '''quantization process
        @params
            z0: {R}^{B, N, d_zx}
                node embeddings
            mask:
                node embedding mask
        @output
            z: {L_i}^{B, N, q_level}
        '''
        _mask = mask.unsqueeze(-1)
        _q_level = torch.tensor(self.q_level, device=z0.device).reshape(1,1,-1)
        # map to quan dim
        z =  self.map_to_quan(z0, mask).tanh()
        z = z * _mask
        
        # quantize
        z = 0.5*_q_level*z
        z_r = torch.round(z)
        
        # rounded ste
        z = z + (z_r - z).detach()
        
        # map to dec dim
        z = self.map_to_dec(z, mask)
        z = z*_mask
        
        return z
    
    @torch.no_grad()
    def sample(self, z0, mask):
        '''
        @params
            z0: {R}^{B, N, d_zx}
                node embedding
            mask:
                node embedding mask
        '''
        _mask = mask.unsqueeze(-1)
        _q_level = torch.tensor(self.q_level, device=z0.device).reshape(1,1,-1)
        # map to quan dim
        z =  self.map_to_quan(z0, mask).tanh()
        z = z*_mask

        # quantize
        z = 0.5*_q_level*z

        z = torch.round(z)
        
        return z
    
    def map_to_quan(self, z, mask):
        '''map to quantization dim
        
        @params
            z: {R}^{B, N, d_zx}
                node embedding
        '''
        return self.enc2q(z, mask)
    
    def map_to_dec(self, z, mask):
        '''map to decoder dim
        
        @params
             z: {L_i}^{B, N, q_level}
        '''
        return self.q2dec(z, mask)
    
class Encoder(nn.Module):
    """
    @params
        in_dim_X: int
            raw feat + extra_feat
        in_dim_y: int
            global extra feat
        in_dim_E: int
            number of high order matrices    
        
    """
    def __init__(self, 
                 in_dim_X,
                 in_dim_y,
                 in_dim_E, 
                 hid_dim_X, hid_dim_E, hid_dim_y,
                 n_layers,
                 n_head):
        super().__init__()
        self.in_dim_E = in_dim_E
        self.hid_dim_X = hid_dim_X

        self.n_layers = n_layers

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(nn.Linear(in_dim_X, hid_dim_X), act_fn_in,
                                      nn.Linear(hid_dim_X, hid_dim_X), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(in_dim_E, hid_dim_E*2), act_fn_in,
                                      nn.Linear(hid_dim_E*2, hid_dim_E), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(in_dim_y, hid_dim_y*2), act_fn_in,
                                      nn.Linear(hid_dim_y*2, hid_dim_y), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hid_dim_X,
                                                            de=hid_dim_E,
                                                            dy=hid_dim_y,
                                                            n_head=n_head,
                                                            dim_ffX=hid_dim_X,
                                                            dim_ffE=hid_dim_E*2)
                                        for _ in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hid_dim_X, hid_dim_X), act_fn_out,
                                       nn.Linear(hid_dim_X, hid_dim_X))

    def forward(self, X, E, node_mask):
        '''
        @params
            X: {R}^(bs, n, 4)
            E: {R}^(bs, n, n)
            node_mask: {0,1}^(bs, n)
            
        @outputs
            X: {R}^(bs, n, hid_dim_X)
        '''  
        adj = E.gt(0.1).float() 
        assert adj.max() == 1.0, pdb.set_trace()
        assert adj.min() == 0.0

        _sup_feat_x, y = add_supp_feat(adj, node_mask.gt(0.1))
        # higher-order adj
        E = pow_tensor(E, self.in_dim_E).permute(0,2,3,1)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        E = mask_adjs(new_E.permute(0,3,1,2), node_mask).permute(0,2,3,1) # bs x n x n x c

        X = torch.cat([X, _sup_feat_x], dim=-1)

        X = mask_x(self.mlp_in_X(X), node_mask)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        X = mask_x(X, node_mask)
        
        return X

class Decoder(nn.Module):
    """
    @params:
        in_dim_X: d_zx
        in_dim_y: d_zx (take a global mean feature)
        out_dim_X: number of classes (without counting the virtual node for molecules)
        
        in_dim_E: number of high order matrices
        out_dim_E: number of high order matrices
    """
    def __init__(self, 
                 in_dim_X,  in_dim_y,
                 out_dim_X,
                 in_dim_E, 
                 hid_dim_X, hid_dim_E, hid_dim_y,
                 n_layers,
                 n_head,
                 scale):
        
        super().__init__()
        
        self.scale = scale
        self.n_layers = n_layers
        self.out_dim_X = out_dim_X
        self.out_dim_E = in_dim_E
        self.in_dim_E = in_dim_E

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(nn.Linear(in_dim_X, hid_dim_X), act_fn_in,
                                      nn.Linear(hid_dim_X, hid_dim_X), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(in_dim_E, hid_dim_E*2), act_fn_in,
                                      nn.Linear(hid_dim_E*2, hid_dim_E), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(in_dim_y, hid_dim_y*2), act_fn_in,
                                      nn.Linear(hid_dim_y*2, hid_dim_y), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hid_dim_X,
                                                            de=hid_dim_E,
                                                            dy=hid_dim_y,
                                                            n_head=n_head,
                                                            dim_ffX=hid_dim_X,
                                                            dim_ffE=hid_dim_E*2)
                                        for _ in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hid_dim_X, hid_dim_X), act_fn_out,
                                       nn.Linear(hid_dim_X, out_dim_X))

        self.mlp_out_E = nn.Sequential(nn.Linear(hid_dim_E, hid_dim_E*2), act_fn_out,
                                       nn.Linear(hid_dim_E*2, self.out_dim_E))

    def forward(self, X, E, y, node_mask):
        '''
        @params
            X: {R}^(bs, n, d_zx)
            E: {R}^(bs, n, n)
            y: {R}^(bs, d_zx)
            node_mask: {0,1}^(bs, n)
            
        @outputs
            X: {R}^(bs, n, num_classes),
                node class probability predition, already softmaxed.
            E: [0,1)^(bs, n, n)
                edge prediction scaled between 0 - 1
        '''
        bs, n = X.shape[0], X.shape[1]  
        # higher-order adj
        E = pow_tensor(E, self.in_dim_E).permute(0,2,3,1)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X
        E_to_out = E

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        E = mask_adjs(new_E.permute(0,3,1,2), node_mask).permute(0,2,3,1) # bs x n x n x c

        X = mask_x(self.mlp_in_X(X), node_mask)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = (X + X_to_out)
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E) 
        E = (E + E_to_out) * diag_mask
        
        E = 1/2 * (E + torch.transpose(E, 1, 2))

        X = torch.nn.functional.softmax(X, dim=-1)

        if self.scale == 3:
            # molecules
            E = torch.sigmoid(E)
            E = E[:,:,:,0] * 1./self.scale + E[:,:,:,1] * 2./self.scale
        
        elif self.scale == 1:
            # spectral graphs
            E = E.sum(-1).sigmoid()
        
        else:
            raise (f'Scale value {self.scale} is not supported')
            
        X = mask_x(X, node_mask) 
        E = mask_adjs(E, node_mask) * diag_mask.squeeze(-1)
        return X, E
       
class DriftModel(nn.Module):
    """Dift Coefficient Model
    @params
        in_dim_X: int
            number of channels L=(5,5,5,5,5) -> in_dim_X: 5
        in_dim_y: int
            time feat + global feat OR  time feat
        in_dim_E: int
            number of high order matrices    
        
    """
    def __init__(self, 
                 in_dim_X,
                 in_dim_y,
                 in_dim_E, 
                 hid_dim_X, hid_dim_E, hid_dim_y,
                 n_layers,
                 n_head):
        super().__init__()
        self.in_dim_E = in_dim_E
        self.out_dim_X = in_dim_X

        self.n_layers = n_layers

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(nn.Linear(in_dim_X, hid_dim_X), act_fn_in,
                                      nn.Linear(hid_dim_X, hid_dim_X), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(in_dim_E, hid_dim_E*2), act_fn_in,
                                      nn.Linear(hid_dim_E*2, hid_dim_E), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(in_dim_y, hid_dim_y*2), act_fn_in,
                                      nn.Linear(hid_dim_y*2, hid_dim_y), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hid_dim_X,
                                                            de=hid_dim_E,
                                                            dy=hid_dim_y,
                                                            n_head=n_head,
                                                            dim_ffX=hid_dim_X,
                                                            dim_ffE=hid_dim_E*2)
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hid_dim_X, hid_dim_X), act_fn_out,
                                       nn.Linear(hid_dim_X,  self.out_dim_X))

    def forward(self, X, E, node_mask, t, y):
        '''
        @params
            X: {R}^(bs, n, d_x)
            E: {R}^(bs, n, n)
            t: {0, ..., N-1}^(bs, 1)
            y: {R}^(bs, d_y)
                optional global embed
            node_mask: {0,1}^(bs, n)
            
        @outputs
            X: {R}^(bs, n, hid_d_x)
        '''
        # add global embed
        if y is not None:
            y = torch.concat([y, t], dim=-1)
        else:
            y = t
        y = y.float()
        # higher-order adj
        E = pow_tensor(E, self.in_dim_E).permute(0,2,3,1)
        X_to_out = X.clone()

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        E = mask_adjs(new_E.permute(0,3,1,2), node_mask).permute(0,2,3,1) # bs x n x n x c

        X = mask_x(self.mlp_in_X(X), node_mask)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        X = (X + X_to_out)
        
        X = mask_x(X, node_mask)
        return X