import os
import pdb
import numpy as np
import shutil
import torch
from torch_geometric.loader import DataLoader
from collections import OrderedDict
from ._base import Logger, Writer
from .train import FSQTrainer, BridgeTrainer
from .loss import FSQLoss, BridgeLoss
from ..data.dataset import GraphDataset
from ..data._add_spec_feat import add_supp_feat
from ..model.model import MolFSQAE, SpecFSQAE, LatentBridge
from ..metric.metric import GenericMetric, Qm9Metric, Zinc250Metric
from .sampler import MolSampler, SpecSampler
from .ema import ExponentialMovingAverage

def _calculate_supp_feat_dim(adj, mask):
    '''calculate supplemenetary feature dim
    @params
        adj: {0, 1}^(1, N, N)
        mask: {False, True}^(1, N)
    '''
    x_feat, y_feat = add_supp_feat(adj.float(), mask)
    return x_feat.shape[-1], y_feat.shape[-1]

def _create_exp_dir(path, scripts_to_save=None):
    '''
    '''
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def _common_init(rank, seed, save_dir, logger_name=None, enable_writer=False):
    '''
    '''
    # we use different seeds per gpu. But we sync the weights after model initialization.
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True

    # prepare logging and tensorboard summary
    logging = Logger(rank, save_dir, logger_name)
    if enable_writer:
        writer = Writer(rank, save_dir)
        
        return logging, writer
    else:
        return logging
    
def load_sampler(n, dataset, **kwargs):
    '''
    @params
        n: int
            number of samples
    '''
    if dataset in ['community', 'ego', 'ENZYMES']:
        sampler = SpecSampler(n, train_loader=kwargs['train_loader'], test_loader=kwargs['val_loader'])
    elif dataset in ['QM9', 'ZINC250k']:
        sampler = MolSampler(n, dataset)
    return sampler
    
def load_metric(dataset):
    '''
    '''
    if dataset in ['community', 'ego', 'ENZYMES']:
        metric = GenericMetric
    elif dataset in ['QM9']:
        metric = Qm9Metric
    elif dataset in ['ZINC250k']:
        metric = Zinc250Metric
    else:
        raise NotImplementedError
    return metric
    
def load_trainer(model_type):
    '''load trainer utils
    @params
        model_type
    '''
    if model_type == 'FSQ':
        trainer = FSQTrainer
        loss_func = FSQLoss
            
    elif model_type == 'Bridge':
        trainer = BridgeTrainer
        loss_func = BridgeLoss
    else:
        raise NotImplementedError
    
    return trainer, loss_func

def load_init_bridge(cfg):
    '''load initialized bridge
    @params
        cfg:
            overall config
    '''
    _L = cfg.model.base.quantizer.q_level
    _N = cfg.model.base.quantizer.max_node_num
    _shape = (_N, len(_L))
    
    _cfg = cfg.train.bridge
    _sig_min = _cfg.sig_min
    _sig_max = _cfg.sig_max
    _noise_type = _cfg.noise_type
    _init_type = _cfg.init_type
    _num_step = _cfg.num_step
    _noise_decay = _cfg.noise_decay
    _use_global_embed = _cfg.use_global_embed
    
    bridge = LatentBridge(_L, _shape, _sig_min, _sig_max,
                    _num_step, _init_type, _noise_type, _noise_decay, _use_global_embed, cfg.model.bridge)
    return bridge
    
def load_init_model(cfg, supp_d_x, supp_d_y):
    '''load initialized moddels
    @params
        cfg: 
            overall config
        supp_d_x: int
            extra x feat dimension
        supp_d_y: int
            extra y feat dimension
    '''
    d_x = cfg.data.d_x
    
    if cfg.data.name in ['ZINC250k', 'QM9']:
        model = MolFSQAE(supp_d_x, supp_d_y, d_x, cfg.model.base, cfg.train.base.scale)
    else:
        model = SpecFSQAE(supp_d_x, supp_d_y, d_x, cfg.model.base, cfg.train.base.scale)
    
    return model

def load_model_weight(model, ckpt_file):
    '''load only model weight
    '''
    ckpt = torch.load(ckpt_file)
    
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()
    for k, val in state_dict.items():
        new_state_dict[k.split('module.')[-1]] = val
    model.load_state_dict(new_state_dict)
    
    return model
    
def load_model_from_ckpt(model, opt, sched, ckpt_file, distributed, decay=0.999):
    '''
    '''
    device = list(range(torch.cuda.device_count()))
    
    ckpt = torch.load(ckpt_file, map_location=f'cuda:{device[0]}')
    cfg = ckpt['cfg']
    cfg.train.is_continue = True
    epoch, global_step, best_score = ckpt['epoch'], ckpt['global_step'], ckpt['best_score']
    
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()
    for k, val in state_dict.items():
        new_state_dict[k.split('module.')[-1]] = val
    model.load_state_dict(new_state_dict)
    
    if distributed:
        model = torch.nn.DataParallel(model, device_ids=device)
    model = model.to(f'cuda:{device[0]}')
    
    opt_dict, sched_dict = ckpt['optimizer'], ckpt['scheduler']
    opt.load_state_dict(opt_dict)
    sched.load_state_dict(sched_dict)
    
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ckpt['ema'])
    
    return cfg, epoch, global_step, best_score, model, opt, sched, ema

def load_logger(save_dir, seed, log_file):
    '''
    '''
    _create_exp_dir(save_dir)
    logger = _common_init(0, seed, save_dir, log_file)
    
    return logger

def load_data(dataset, proj_dir, batch_size, logger=None):
    '''
    '''
    train_data = GraphDataset(dataset, proj_dir + '/_processed_data/', train=True)[0]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_data = GraphDataset(dataset, proj_dir + '/_processed_data/', train=False)[0]
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    
    # calculate extra feat
    if dataset in ['community', 'ego', 'ENZYMES']:
        supp_d_x, supp_d_y = _calculate_supp_feat_dim(train_data[0][1].unsqueeze(0), train_data[0][1].unsqueeze(0).sum(-1).gt(0.1))
    else:
        supp_d_x, supp_d_y =  _calculate_supp_feat_dim(train_data[0][1].unsqueeze(0).gt(0.1).long(), train_data[0][1].unsqueeze(0).sum(-1).gt(0.1))
    
    if logger is not None:
        logger.info('load data: %d train, %d val', len(train_data), len(val_data))
    return train_loader, val_loader, supp_d_x, supp_d_y

def load_optimizer(model, cfg, gamma=0.999, amsgrad=True):
    '''
    @params
        cfg:
            train cfg
    '''
    optimizer =  torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay, amsgrad=amsgrad)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return optimizer, scheduler

def load_optimizer_bridge(model, cfg, amsgrad=True):
    '''
    @params
        cfg:
            train bridge cfg
    '''
    optimizer =  torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay, amsgrad=amsgrad)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(cfg.num_epoch - cfg.warmup_epoch - 1), eta_min=cfg.lr_min)

    return optimizer, scheduler

def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema