import pdb
import os, sys
import time
import torch
import logging
import torch.nn as nn
from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter
 
class TrainerBase(ABC):
    '''
    @params
        cfg:
            overall cfg
        global_step: int
            global step.
        best_score: float
            best score so far.
        init_epoch: int
            init epoch.
        num_epoch: int
            total training epoch
        num_eval: int
            number evaluations
    '''
    def __init__(self, save_dir, cfg, num_eval, num_epoch, model, opt, sched, loss_func, logger, 
                    global_step, best_score, init_epoch, sampler):
        super(TrainerBase, self).__init__()
        self.save_dir = save_dir
        self.cfg = cfg
        self.num_eval = num_eval
        self.num_epoch = num_epoch
        self.model = model
        self.opt = opt
        self.sched = sched
        self.logger = logger
        self.sampler = sampler
        self.loss_func = loss_func
        self.global_step = 0 if global_step is None else global_step 
        self.init_epoch = 0 if init_epoch is None else init_epoch 
        self.best_score = 0 if best_score is None else best_score
        
    @abstractmethod
    def train_step(self):
        pass
    
    @abstractmethod
    def val_step(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def eval(self):
        pass
    
    def save(self, epoch, cfg, ckpt_file):
        '''
        '''
        if self.ema is not None:
            content = {'epoch': epoch + 1, 'global_step': self.global_step, 'cfg': cfg,
                    'best_score': self.best_score,
                    'state_dict': self.model.state_dict(), 'optimizer': self.opt.state_dict(),
                    'scheduler': self.sched.state_dict(),
                    'ema': self.ema.state_dict()}
        else:
            content = {'epoch': epoch + 1, 'global_step': self.global_step, 'cfg': cfg,
                    'best_score': self.best_score,
                    'state_dict': self.model.state_dict(), 'optimizer': self.opt.state_dict(),
                    'scheduler': self.sched.state_dict()}
        torch.save(content, ckpt_file)

class SamplerBase(ABC):
    '''
    @params 
        n: int
            number of samples
    '''
    def __init__(self, n):
        self.n = n

    @abstractmethod
    def sample(self):
        pass
            

class Logger(object):
    def __init__(self, rank, save, name):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            if name is None:
                fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            else:
                fh = logging.FileHandler(os.path.join(save, name))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)

class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:  # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()
