import pdb
import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from src import load_data, load_logger, load_model_weight, \
        count_parameters_in_M, load_init_model, load_init_bridge, load_sampler 

@hydra.main(version_base='1.1', config_path='./config', config_name='config')
def main(cfg: DictConfig):

    # init folder + logger
    save_root = cfg.train.save_root
    seed =  cfg.train.seed
    log_file = 'sample_' + cfg.train.log_file
    proj_dir = os.getcwd().split('outputs')[0]
    save_dir = proj_dir + save_root + cfg.train.save_dir
    base_dir = proj_dir + save_root + cfg.train.base_dir
    logger = load_logger(save_dir, seed, log_file)
    logger.info('args = %s', cfg)
    
    # load data
    _dataset = cfg.data.name
    if _dataset == 'QM9':
        _batch_size = 10000
        _N = 10000
    elif _dataset == 'ZINC250k':
        _batch_size = 2000
        _N = 10000
    else:
        _batch_size = cfg.train.sampler.num_sample
        _N = cfg.train.sampler.num_sample
    train_loader, val_loader, supp_d_x, supp_d_y = load_data(_dataset, proj_dir, _batch_size, logger)
    
    # load model
    model = load_init_model(cfg, supp_d_x, supp_d_y)
    logger.info('MODEL: param size = %fM ', count_parameters_in_M(model))
    
    # load best model
    if cfg.train.bridge.load_best_base:
        _ckpt_file = os.path.join(base_dir, 'best_ckpt.pt')
        logger.info('Loaded best model base.')
    else:
        _ckpt_file = os.path.join(base_dir, 'last_ckpt.pt')
        logger.info('Loaded last model base ...')
    model = load_model_weight(model, _ckpt_file)
    if torch.cuda.device_count() > 1:
        m_device, p_device = 'cuda:0', 'cuda:1'
    else:
        m_device, p_device = 'cuda:0', 'cuda:0'
        
    model = model.to(m_device)
    model.eval()

    for bridge_ckpt in ['bridge_best_ckpt.pt', 'bridge_last_ckpt.pt']:
        bridge = load_init_bridge(cfg)
        logger.info('BRIDGE: param size = %fM ', count_parameters_in_M(bridge.model))
        
        bridge_ckpt_file = os.path.join(save_dir, bridge_ckpt)
        bridge.model = load_model_weight(bridge.model, bridge_ckpt_file)
        bridge.model = bridge.model.to(p_device)
        bridge.model.eval()
        logger.info('Evaluate %s', bridge_ckpt)
        # evaluate graph generation
        sampler = load_sampler(_N, _dataset, train_loader=train_loader, val_loader=val_loader)
        logger.info(f'start sampling {bridge_ckpt}...')
        if cfg.data.name in ['QM9', 'ZINC250k']:
            result_dict = {'valid_wo_correct': [],
                           'valid': [], 'unique@10000': [],
                           'FCD/Test': [], 'Novelty': [], 'nspdk_mmd': []}
            for _ in range(3):
                score, result = sampler.sample(logger, model, bridge, train_loader)
                for k, v in result.items():
                    result_dict[k].append(v)
            for key, val in result_dict.items():
                logger.info('%s:  %.5f +- %.5f', key, np.mean(val), np.std(val))
        else:
            result_dict = {'degree': [],
                        'clustering': [],
                        'orbit': []}
            for _ in range(15):
                score, result = sampler.sample(logger, model, bridge, train_loader)
                for key, val in result.items():
                    result_dict[key].append(val)
            mean_all = []
            for key, val in result_dict.items():
                logger.info('%s:  %.5f +- %.5f', key, np.mean(val), np.std(val))
                mean_all.append(np.mean(val))
            logger.info('mean-all:  %.5f', np.mean(mean_all))
        logger.info('DONE !!!')
if __name__ == '__main__':
    main()