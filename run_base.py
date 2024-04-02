import pdb
import os
import torch
import hydra
from omegaconf import DictConfig
from src import load_data, load_logger, load_optimizer, \
            load_model_from_ckpt, count_parameters_in_M, load_ema, load_init_model, load_trainer

@hydra.main(version_base='1.1', config_path='./config', config_name='config')
def main(cfg: DictConfig):
    # init folder + logger
    save_root = cfg.train.save_root
    seed =  cfg.train.seed
    log_file = 'base_' + cfg.train.log_file
    proj_dir = os.getcwd().split('outputs')[0]
    save_dir = proj_dir + save_root + cfg.train.save_dir
    logger = load_logger(save_dir, seed, log_file)
    logger.info('args = %s', cfg)
    
    # load data
    _dataset = cfg.data.name
    _batch_size = cfg.train.batch_size
    train_loader, val_loader, supp_d_x, supp_d_y = load_data(_dataset, proj_dir, _batch_size, logger)
    
    # load model
    model = load_init_model(cfg, supp_d_x, supp_d_y)
    logger.info('MODEL: param size = %f M ', count_parameters_in_M(model))
    opt, sched = load_optimizer(model, cfg.train.base)
    
    _distributed = cfg.train.distributed
    device = list(range(torch.cuda.device_count()))
    # init global vars
    global_step, init_epoch, best_score = 0, 0, 0
    
    if cfg.train.is_continue:
        last_ckpt_file = os.path.join(save_dir, 'last_ckpt.pt')
        cfg, init_epoch, global_step, best_score, model, opt, sched, ema = load_model_from_ckpt(model, opt, sched, last_ckpt_file, _distributed)
    else:
        if _distributed:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
        ema = load_ema(model)
        
    _cfg = cfg.train.base
    num_eval = _cfg.num_eval
    num_epoch = _cfg.num_epoch
    scale = _cfg.scale
    clip_norm = _cfg.clip_norm
    
    Trainer, Loss = load_trainer('FSQ')
    loss_func = Loss(scale)
    trainer = Trainer(save_dir, cfg, num_eval, num_epoch, model, opt, sched, loss_func, logger, 
                    global_step, best_score, init_epoch, sampler= None, ema=ema)
    
    trainer.train(train_loader, val_loader, clip_norm=clip_norm)
    
if __name__ == '__main__':
    main()