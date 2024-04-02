import pdb
import torch
import os
import hydra
from omegaconf import DictConfig
from src import load_data, load_logger, load_optimizer_bridge, load_sampler, load_trainer,\
            load_model_from_ckpt, count_parameters_in_M, load_ema, load_model_weight, load_init_model, load_init_bridge

@hydra.main(version_base='1.1', config_path='./config', config_name='config')
def main(cfg: DictConfig):

    # init folder + logger
    save_root = cfg.train.save_root
    seed =  cfg.train.seed
    log_file = 'bridge_' + cfg.train.log_file
    proj_dir = os.getcwd().split('outputs')[0]
    save_dir = proj_dir + save_root + cfg.train.save_dir
    base_dir = proj_dir + save_root + cfg.train.base_dir
    logger = load_logger(save_dir, seed, log_file)
    logger.info('args = %s', cfg)
    new_epoch = cfg.train.bridge.num_epoch
    
    # load data
    _dataset = cfg.data.name
    _batch_size = cfg.train.batch_size
    train_loader, val_loader, supp_d_x, supp_d_y = load_data(_dataset, proj_dir, _batch_size, logger)
    
    # load model
    model = load_init_model(cfg, supp_d_x, supp_d_y)
    bridge = load_init_bridge(cfg)
    
    logger.info('MODEL: param size = %f M ', count_parameters_in_M(model))
    logger.info('BRIDGE: param size = %fM ', count_parameters_in_M(bridge.model))
    opt, sched = load_optimizer_bridge(bridge.model, cfg.train.bridge)
    
    _distributed = cfg.train.distributed
    device = list(range(torch.cuda.device_count()))
    # load best model
    if cfg.train.bridge.load_best_base:
        _ckpt_file = os.path.join(base_dir, 'best_ckpt.pt')
        logger.info('Loaded best model base.')
    else:
        _ckpt_file = os.path.join(base_dir, 'last_ckpt.pt')
        logger.info('Loaded last model base ...')
    model = load_model_weight(model, _ckpt_file)
    
    if _distributed:
        model = torch.nn.DataParallel(model, device_ids=device)
    model = model.to(f'cuda:{device[0]}')
    
    # init global vars
    global_step, init_epoch, best_score = 0, 0, -1.0
    if cfg.train.is_continue:
        last_ckpt_file = os.path.join(save_dir, 'bridge_last_ckpt.pt')
        cfg, init_epoch, global_step, best_score, bridge.model, opt, sched, ema = load_model_from_ckpt(bridge.model, opt, sched, last_ckpt_file, _distributed)
    else:
        if _distributed:
            bridge.model = torch.nn.DataParallel(bridge.model, device_ids=device)
        bridge.model = bridge.model.to(f'cuda:{device[0]}')
        ema = load_ema(bridge.model)
            
    # load sampler
    _N = cfg.train.sampler.num_sample
    sampler = load_sampler(_N, _dataset, train_loader=train_loader, val_loader=val_loader)
    
    _cfg = cfg.train
    num_eval = _cfg.bridge.num_eval
    num_epoch = _cfg.bridge.num_epoch if _cfg.bridge.num_epoch == new_epoch else new_epoch
    warmup_epoch = _cfg.bridge.warmup_epoch

    Trainer, Loss = load_trainer('Bridge')

    loss_func = Loss()
    trainer = Trainer(bridge, save_dir, cfg, num_eval, num_epoch, model, opt, sched, loss_func, logger, 
                    global_step, best_score, init_epoch, sampler=sampler, ema=ema, warmup_epoch=warmup_epoch)
    _clip_norm = _cfg.bridge.clip_norm
    trainer.train(train_loader, val_loader, distributed= _distributed, clip_norm= _clip_norm)
    
if __name__ == '__main__':
    main()