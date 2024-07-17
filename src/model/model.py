import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from operator import mul
from functools import reduce
from ._lib import build_mask
from ._base import Base
from ._module import FSQuantizer, DriftModel

class FSQAE(Base):
    '''Finite Scalar Quantization Graph AutoEncoder.
    @params
        d_raw_x: int
            original x feat
        supp_d_x: int
            supp x feat
        supp_d_y: int
            y feat
        scale: int
            edge scaling coeff to [0,1)
        cfg: base model config 
    '''
    def __init__(self, supp_d_x, supp_d_y,  d_raw_x, scale, cfg):
        super().__init__(supp_d_x, supp_d_y,  d_raw_x, scale, cfg)
        if cfg.quantizer.act == 'selu':
            act = nn.SELU()
        elif cfg.quantizer.act == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f'Act {cfg.quantizer.act} is not supported.')
        self.quantizer = FSQuantizer(cfg.quantizer.q_level, self.enc.hid_dim_X, 
                            cfg.quantizer.max_node_num, act=act, depth=cfg.quantizer.num_layers,
                            is_norm=cfg.quantizer.is_norm)
        
    def forward(self, x_in, e_in, sampling=False, add_noise=False):
        '''
        @params
            x_in: {0, 1}^(B, N, d_x)
                x: input feat
                d_x: number of node classes
            e_in: {0, ..., d_e}^(B, N, N)
                e: input feat
                d_e: number of edge bond types
            sampling: bool
                if only sample the latent embeddings
        @return
            x_out: {R}^(B, N, d_x)
            e_out: {[0, 1)}^(B, N, N)
        '''
        mask = build_mask(e_in)
        e_in = e_in.float()/self.scale # scale adj matrix
        
        if add_noise:
        # add noise to break symmetry problem
            x_in = x_in + 0.1*torch.randn_like(x_in)
            x_in = x_in*mask.unsqueeze(-1)
        
        z = self.enc(x_in, e_in, mask)
        
        if sampling:
            return self.quantizer.sample(z, mask)
        # quantize and map2dec
        z = self.quantizer(z, mask)
        # decoding
        x_out, e_out = self._decode(z, mask)
        return x_out, e_out
    
    def _decode(self, z, mask):
        '''decoding
        @params 
            z: map2dec from quantizer
            mask: node masking
        '''
        # global pooling
        y_dec = z.mean(1)
        # latent adj
        adj_dec = torch.bmm(z, z.transpose(1,2)).sigmoid()
        # masking
        adj_dec = adj_dec * mask.unsqueeze(1) * mask.unsqueeze(2)
        # sym
        adj_dec = 1/2 * (adj_dec + adj_dec.transpose(1,2))
        # diag
        adj_dec = adj_dec * (~torch.eye(adj_dec.shape[-1], device=adj_dec.device).gt(0).unsqueeze(0)).float()
        
        x_out, e_out = self.dec(z, adj_dec, y_dec, mask)
        return x_out, e_out
        
    
class LatentBridge(object):
    '''Latent Bridge
    
    @params:
        L: list
            quantization levels
        data_shape: tuple
            (N, F)
        use_global_embed: bool
            use global embedding as y feature
        cfg:
            bridge model config
    '''
    def __init__(self, 
                 L,
                 data_shape,
                 sig_min,
                 sig_max,
                 N,
                 init_type,
                 noise_type,
                 noise_decay, 
                 use_global_embed,
                 cfg): 
        self.data_shape = data_shape
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.N = N
        self.L = L
        self.init_type = init_type
        self.noise_type = noise_type
        self.noise_decay = noise_decay
        self.use_global_embed = use_global_embed
        self.codebook = self._get_codebook(L)
        self.noise_param = self._get_noise(noise_type)
        
        if use_global_embed:
            in_dim_y = data_shape[-1] + 1 
        else:
            in_dim_y = 1 
        
        in_dim_X = data_shape[-1]
        in_dim_E = cfg.in_dim_E
        hid_dim_X = cfg.hid_dim_X
        hid_dim_E = cfg.hid_dim_E
        hid_dim_y = cfg.hid_dim_y
        n_layers = cfg.n_layers
        n_head = cfg.n_head

        self.model = DriftModel(in_dim_X,
                                in_dim_y,
                                in_dim_E, 
                                hid_dim_X, hid_dim_E, hid_dim_y,
                                n_layers,
                                n_head)
    
    def _get_x0(self, bs, device, mask):
        '''
        @params
            bs: int
                batch size
            mask:
                {0,1} ^ (B, N)
        '''
        _shape = self.data_shape
        _type = self.init_type
        _bshape = (bs, _shape[0], _shape[1])
        
        if _type == 'const':
            x0 = torch.zeros(_bshape, device=device)
            
        elif _type == 'gaussian':
            x0 = torch.randn(_bshape, device=device)
            if mask is not None:
                x0 =  x0 * mask.unsqueeze(-1)
             
        else:
            raise NotImplementedError(f'{_type} not implemented.')
        
        return x0
            
    def _get_noise(self, noise_type):
        if noise_type == 'exp':
            t = torch.linspace(0, 1.-1./self.N, self.N)
            sigma = self.sig_min * self.sig_max * torch.exp((-1.) * self.sig_max * t)
            beta = (-1.) * self.sig_min * torch.exp((-1.) * self.sig_max * t)
            bias = beta[0]
            beta = beta - bias
            beta_T = (-1.) * self.sig_min * torch.exp((-1.) * self.sig_max * torch.Tensor([1.])) - bias
            denominator = beta_T - beta
        else:
            raise NotImplementedError(f'{noise_type} not implemented.')
        
        return [sigma, denominator, beta_T, beta]
    
    def _get_codebook(self, L: list):
        '''Implement horizontal-dim codebooks.
        @params
            L: list
                different quantization levels.
        '''
        cb_level = []
        for i in range(len(L)):
            cb_level.append([j - L[i] // 2 for j in range(L[i])])
        cb_perm = list(itertools.product(*cb_level))
        assert len(cb_perm) == reduce(mul, L)
        cb_tensor = torch.Tensor(cb_perm).unsqueeze(0).unsqueeze(0) # (1, 1, K, F); K: cbook size.
        return cb_tensor
    
    @property
    def T(self):
        return 1.
    
    def _forward(self, z0, mask):
        '''
        @params
            z0: {L_0, ..., L_n}^(B, N, F)
                quantized embedding
            mask: {0,1}^(B, N)
                node mask
        '''
        perb_z, obj, t_sample = self._get_pertubed_data(z0, mask)
        perb_z = perb_z * mask.unsqueeze(-1)
        obj = obj *  mask.unsqueeze(-1)
        
        y, adj = self._get_input(perb_z, mask)
        pred = self.model(perb_z, adj, mask, t_sample, y)

        return obj, pred
    
    def _get_input(self, perb_z, mask):
        '''get input to denoising network
        '''
        adj = torch.bmm(perb_z, perb_z.transpose(1,2)).sigmoid()
        adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        adj = 1/2 * (adj + adj.transpose(1,2))
        adj = adj * (~torch.eye(adj.shape[-1], device=adj.device).gt(0).unsqueeze(0)).float()
        
        if self.use_global_embed:
            perb_global_z = perb_z.mean(1)
        else:
            perb_global_z = None

        return perb_global_z, adj
    
    @torch.no_grad()
    def _get_pertubed_data(self, batch, mask):
        '''Sample from the intermediate conditional dists
        
        @params
            batch: {R}^(B, N, F)
        '''
        _bs = batch.shape[0]
        _device = batch.device
        x0 = self._get_x0(_bs, _device, mask).clone() # (bs, N, F)
        t = torch.randint(0, self.N, (_bs,), device=_device) # (bs)
        
        noise = torch.randn_like(x0, device= _device)
        
        cb = self.codebook.to(device=_device)
        if self.noise_decay:
            # change-variance sampling
            sigma, beta_T, beta = self.noise_param[0], self.noise_param[2], self.noise_param[3] # (N), (1), (N)
            _sigma = sigma[t].to(_device)
            _beta = beta[t].to(_device)
            _beta_T = beta_T.to(_device)
            beta_T_minus_beta = _beta_T - _beta # (bs)
            _beta = _beta.reshape(_bs, 1, 1) # (bs, 1, 1)
            perb_x = (_beta * batch + (_beta_T - _beta) * x0)/_beta_T
            perb_x = perb_x + noise * torch.sqrt(_beta * (1. - _beta / _beta_T)) # (bs, N, F)
            denominator = beta_T_minus_beta.clone() / _sigma
        else:
            raise NotImplementedError('ONLY NOISE-DECAY IMPLEMENTED.')
        
        eta_x = (batch - perb_x).detach().clone() / denominator.reshape(_bs, 1, 1) # (bs, N, F)
        
        with torch.enable_grad():
            z = perb_x.detach().requires_grad_(True)
            log_h_sum = self._get_log_h(cb, z, beta_T_minus_beta, mask)
            grad_log_h = torch.autograd.grad(log_h_sum, z, allow_unused=True, retain_graph=False)[0] # (bs, N, F)
        
        eta_domain = _sigma[:, None, None] * grad_log_h
        eta_obj = eta_x - eta_domain # (bs, N, F)
        t = t.view(-1, 1) # (bs, 1)
        return perb_x, eta_obj, t
        
    def _get_log_h(self, cb, z, beta_T_minus_beta, mask):
        '''Get log distance
        
        @params
            cb: {{L_i}}^(1, 1, K, F)
                codebook
            z: {R}^(B, N, F)
                conditional sampled noisy data
            beta_T_minus_beta: (B)
            mask: {0, 1} ^ (B, N)

        '''
        log_h = (-1) * (z.unsqueeze(2) - cb).norm(dim=-1).pow(2) / (2.*beta_T_minus_beta[:, None, None])  # (B, N, K)
        log_h_max = torch.max(log_h, dim=-1)[0].unsqueeze(-1) # (B, N, 1)
        log_h = log_h - log_h_max.detach()
        log_h = log_h.exp()
        log_h = log_h.sum(dim=-1) # (B, N)

        return log_h.log().sum(-1).sum(-1)
    
    @torch.no_grad()
    def sample(self, n, device, mask, recursive=True):
        '''
        @params:
            n: int
                number of samples.
            mask:
                sample from the training data.
            recursive: bool
                compute drift recursively on each element
        '''
        # create an instance to speed up inference.
        cb = self.codebook.to(device=device)
        
        _mask = mask.unsqueeze(-1)
        x0 = self._get_x0(n, device, mask) # (n, N, F)
        z = x0.clone()
        eps = torch.Tensor([1 / self.N]).to(device)
        
        for t in range(self.N):
            
            noise = torch.randn_like(z, device=device)
            if self.noise_decay:
                sigma, beta_T, beta = self.noise_param[0], self.noise_param[2], self.noise_param[3] # (N), (1), (N)
                _beta = beta[t]
                _sigma = sigma[t]
                beta_T_minus_beta = (beta_T[0] - _beta) * torch.ones((n), device=mask.device)
                noise = torch.sqrt(_sigma) * noise 
            else:
                raise NotImplementedError('ONLY NOISE-DECAY IMPLEMENTED.')

            t = torch.ones((n), device= device) * t
            
            grad_log_h = torch.zeros(z.shape, device=z.device)
            with torch.enable_grad():
                z = z.detach().requires_grad_(True)
                if recursive:
                    for i in range(z.shape[1]):
                        z_i = z[:,i:i+1,:]
                        log_h_sum = self._get_log_h(cb, z_i, beta_T_minus_beta, mask)
                        grad_log_h_i = torch.autograd.grad(log_h_sum, z_i, allow_unused=True, retain_graph=False)[0]
                        grad_log_h[:,i:i+1,:] = grad_log_h_i
                else:
                    log_h_sum = self._get_log_h(cb, z, beta_T_minus_beta, mask)
                    grad_log_h = torch.autograd.grad(log_h_sum, z, allow_unused=True, retain_graph=False)[0] # (bs, N, F)
            t = t.view(-1, 1) # (n, 1)

            with torch.no_grad():
                z = z * _mask
                y, adj = self._get_input(z, mask)
                pred = self.model(z, adj, mask, t, y)
            z = z.detach().clone() + eps * (pred + _sigma * grad_log_h) + torch.sqrt(eps) * noise
            
        z = z * _mask
        z_round = torch.round(z)
        # find approximate point
        z_round = self._clamp_x(z_round)
        return z_round
    
    def _clamp_x(self, x):
        '''clamp tensor values to _min, _max
        '''
        assert sum(self.L) == self.L[0]*len(self.L) 
        
        _min = - (self.L[0] // 2)
        _max = self.L[0] // 2
        x = torch.clamp(x, min=_min, max=_max)
        return x
    
    
class MolFSQAE(FSQAE):
    def __init__(self, supp_d_x, supp_d_y,  d_raw_x, cfg, scale):
        super(MolFSQAE, self).__init__(supp_d_x, supp_d_y,  d_raw_x, scale, cfg)
        
class SpecFSQAE(FSQAE):
    def __init__(self,  supp_d_x, supp_d_y,  d_raw_x, cfg, scale):
        super(SpecFSQAE, self).__init__(supp_d_x, supp_d_y,  d_raw_x, scale, cfg)