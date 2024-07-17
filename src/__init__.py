import numpy as np
from .utils.loader import load_model_from_ckpt, load_logger, load_data, load_optimizer, \
                                load_init_model, load_trainer, load_init_bridge, load_model_weight, load_sampler, load_ema, load_optimizer_bridge

def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

__all__ = [load_trainer, load_model_from_ckpt, load_init_model, load_init_bridge,
        load_logger, load_data, load_optimizer, count_parameters_in_M, load_model_weight, load_sampler, load_ema, load_optimizer_bridge]