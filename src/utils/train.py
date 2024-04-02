import pdb
import os
import torch
from ._base import TrainerBase
from ..model._lib import build_mask

def rand_perm(x, adj):
    flags = build_mask(adj)
    batch_size = adj.shape[0]
    num_nodes = adj.shape[-1]
    idx = torch.arange(num_nodes, device=adj.device).repeat(batch_size, 1)
    for _ in range(batch_size):
        pidx = torch.where(flags[_]>0)[0]
        idx[_][pidx] = idx[_][pidx[torch.randperm(int(flags[_].sum(-1)))]]
    peye = torch.eye(num_nodes, device=adj.device)[idx]
    px = torch.bmm(peye, x)
    padj = torch.bmm(torch.bmm(peye, adj), peye.transpose(-1,-2))
    return px, padj

class FSQTrainer(TrainerBase):
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
                    global_step= None, best_score= None, init_epoch= None, sampler= None, ema=None):
        super(FSQTrainer, self).__init__(save_dir, cfg, num_eval, num_epoch, model, opt, sched, loss_func, logger, 
                    global_step, best_score, init_epoch, sampler)
        self.ema = ema

    def train_step(self, batch, clip_norm):
        '''
         @params
            x_in: {0, 1}^(B, N, d_x)
                x input feat
            e_in: {0, ..., d_e}^(B, N, N)
                e input feat
            clip_norm: bool
        '''     
        self.opt.zero_grad()
        x_in, e_in = batch[0].float().cuda(), batch[1].float().cuda()
        x_in, e_in = rand_perm(x_in, e_in)
        x_out, e_out = self.model(x_in, e_in, add_noise=self.cfg.model.base.add_noise)
        
        loss, e_acc, x_acc = self.loss_func(x_in, e_in, x_out, e_out)
        
        if loss.shape.__len__() > 0:
            loss, e_acc, x_acc = [torch.mean(x) for x in [loss, e_acc, x_acc]]
        
        loss.backward()
        
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        
        self.opt.step()
        # -------- EMA update --------
        if self.ema is not None:
            self.ema.update(self.model.parameters())
            
        self.global_step += 1
        return loss, e_acc, x_acc

    def train(self, train_loader, val_loader, clip_norm):
        '''
        @params
            train_loader:
            val_loader:
        '''
        best_ckpt_file = os.path.join(self.save_dir, 'best_ckpt.pt')
        last_ckpt_file =  os.path.join(self.save_dir, 'last_ckpt.pt')

        for epoch in range(self.init_epoch, self.num_epoch):
            x_acc_meter = []
            e_acc_meter = []
            loss_meter = []

            self.logger.info('epoch: %d', epoch)
            
            self.model.train()
            for batch in train_loader:
                loss, e_acc, x_acc = self.train_step(batch, clip_norm)
                x_acc_meter.append(x_acc.data)
                e_acc_meter.append(e_acc.data)
                loss_meter.append(loss.data)
            self.sched.step()
            
            train_x_acc = sum(x_acc_meter)/len(x_acc_meter)
            train_e_acc = sum(e_acc_meter)/len(e_acc_meter)
            train_loss = sum(loss_meter)/len(loss_meter)
            self.logger.info('train: loss %f, e_acc %f, x_acc %f', train_loss, train_e_acc, train_x_acc)

            eval_freq = max(self.num_epoch // self.num_eval, 1)
            if ((epoch + 1) % eval_freq == 0 or epoch == (self.num_epoch - 1)):
                
                self.model.eval()
                val_loss, val_e_acc, val_x_acc = self.eval(val_loader)
                self.logger.info('val: loss %f, e_acc %f, x_acc %f', val_loss, val_e_acc, val_x_acc)
                
                score = 1. * val_x_acc*val_e_acc
                if score > self.best_score:
                    self.best_score = 1. * score
                    self.save(epoch, self.cfg, best_ckpt_file)

                    self.logger.info('saving the best model.')
                    self.logger.info('best metric result: ' + str(self.best_score))

            self.save(epoch, self.cfg, last_ckpt_file)
        self.logger.info('best metric result: ' + str(self.best_score))

    def val_step(self, batch):
        '''
         @params
            x_in: {0, 1}^(B, N, d_x)
                x input feat
            e_in: {0, ..., d_e}^(B, N, N)
                e input feat
        '''
        x_in, e_in = batch[0].cuda(), batch[1].cuda()
        with torch.no_grad():
            x_out, e_out = self.model(x_in, e_in)
            loss, e_acc, x_acc = self.loss_func(x_in, e_in, x_out, e_out)
            if loss.shape.__len__() > 0:
                loss, e_acc, x_acc = [torch.mean(x) for x in [loss, e_acc, x_acc]]
        return loss, e_acc, x_acc

    def eval(self, val_loader):
        '''
        '''
        x_acc_meter = []
        e_acc_meter = []
        loss_meter = []

        for batch in val_loader:
            _loss, _e_acc, _x_acc = self.val_step(batch)
            x_acc_meter.append(_x_acc.data)
            e_acc_meter.append(_e_acc.data)
            loss_meter.append(_loss.data)

        x_acc, e_acc = sum(x_acc_meter)/len(x_acc_meter), sum(e_acc_meter)/len(e_acc_meter)
        loss = sum(loss_meter)/len(loss_meter)
        return loss, e_acc, x_acc     
    
    
class BridgeTrainer(TrainerBase):
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
        model: ae | vae
            trained ae | vae
        prior:
            bridge prior
    '''
    def __init__(self, prior, save_dir, cfg, num_eval, num_epoch, model, opt, sched, loss_func, logger, 
                    global_step= None, best_score= None, init_epoch= None, sampler= None, ema=None, warmup_epoch=None):
        super(BridgeTrainer, self).__init__(save_dir, cfg, num_eval, num_epoch, model, opt, sched, loss_func, logger, 
                    global_step, best_score, init_epoch, sampler)
        self.prior = prior
        self.ema = ema
        self.warmup_epoch = warmup_epoch
        
    def train_step(self, batch, clip_norm):
        '''
         @params
            x_in: {0, ..., d_x-1}^(B, N)
                x input feat
            e_in: {0, ..., d_e-1}^(B, N, N)
                e input feat
            clip_norm: bool
        '''
        
        if self.global_step < self.warmup_iter:
            lr = self.cfg.train.bridge.lr * self.global_step / self.warmup_iter
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
        
        self.opt.zero_grad()  
        x_in, e_in = batch[0].float().cuda(), batch[1].float().cuda()
        x_in, e_in = rand_perm(x_in, e_in)
        mask = build_mask(e_in)
        
        with torch.no_grad():
            z0 = self.model(x_in, e_in, sampling=True, add_noise=self.cfg.model.base.add_noise)
        obj, pred = self.prior._forward(z0, mask)
        # distance loss
        loss = self.loss_func(obj, pred)
        loss.backward()
        
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.prior.model.parameters(), 5)
        
        self.opt.step()
        # -------- EMA update --------
        if self.ema is not None:
            self.ema.update(self.prior.model.parameters())
        
        self.global_step += 1
        return loss
    
    def val_step(self, batch):
        '''
         @params
            x_in: {0, ..., d_x-1}^(B, N)
                x input feat
            e_in: {0, ..., d_e-1}^(B, N, N)
                e input feat
        '''
        x_in, e_in = batch[0].cuda(), batch[1].cuda()
        mask = build_mask(e_in)
            
        with torch.no_grad():
            z0 = self.model(x_in=x_in, e_in=e_in, sampling=True)
            obj, pred = self.prior._forward(z0, mask)
            # distance loss
            loss = self.loss_func(obj, pred)
        return loss
        
    def train(self, train_loader, val_loader, distributed, clip_norm):
        '''
        @params
            train_loader:
            val_loader:
            distributed:
            clip_norm: bool
                gradient clipping
            masking: bool
                apply node masking
        '''
        self.warmup_iter = len(train_loader) * self.warmup_epoch
        best_ckpt_file = os.path.join(self.save_dir, 'bridge_best_ckpt.pt')
        last_ckpt_file =  os.path.join(self.save_dir, 'bridge_last_ckpt.pt')

        self.model.eval()
        for epoch in range(self.init_epoch, self.num_epoch):
            loss_meter = []
            
            self.logger.info('epoch: %d', epoch)

            self.prior.model.train()
            for batch in train_loader:
                loss = self.train_step(batch, clip_norm)
                loss_meter.append(loss.data)
            
            if epoch > self.warmup_epoch:
                self.sched.step()
            
            loss = sum(loss_meter) / len(loss_meter)
            self.logger.info('train: loss %f', loss)
            
            eval_freq = max(self.num_epoch // self.num_eval, 1)
            if ((epoch + 1) % eval_freq == 0 or epoch == (self.num_epoch - 1)):
    
                self.prior.model.eval()
                loss = self.eval(val_loader)
                score = self.gen_stats(train_loader, self.logger, distributed)
                self.logger.info('val: %f, score: %f', loss, score)
                score = 1. * score
                if score > self.best_score:
                    self.best_score = 1. * score
                    self.save(epoch, self.cfg, best_ckpt_file)

                    self.logger.info('saving the best model.')
                    self.logger.info('best nvu metric result: ' + str(self.best_score))

            self.save(epoch, self.cfg, last_ckpt_file)
        self.logger.info('best nvu metric result: ' + str(self.best_score))
    
    def eval(self, val_loader):
        '''
        '''
        loss_meter = []
        for batch in val_loader:
            loss = self.val_step(batch)
            loss_meter.append(loss.data)
            
        loss = sum(loss_meter)/len(loss_meter)
        return loss
    
    def save(self, epoch, cfg, ckpt_file):
        '''
        '''
        # -------- EMA update --------
        if self.ema is not None:
            content = {'epoch': epoch + 1, 'global_step': self.global_step, 'cfg': cfg,
                    'best_score': self.best_score,
                    'state_dict': self.prior.model.state_dict(), 'optimizer': self.opt.state_dict(),
                    'scheduler': self.sched.state_dict(),
                    'ema': self.ema.state_dict()}
        else:
            content = {'epoch': epoch + 1, 'global_step': self.global_step, 'cfg': cfg,
                    'best_score': self.best_score,
                    'state_dict': self.prior.model.state_dict(), 'optimizer': self.opt.state_dict(),
                    'scheduler': self.sched.state_dict()}
        torch.save(content, ckpt_file)
        
    def gen_stats(self, train_loader, logger, distributed):
        '''intermediate sampled result
        '''
        if distributed:
            score, _ = self.sampler.sample(logger, self.model.module, self.prior, train_loader) #valid*unique
                                                                                            #mean(degress,orbit,clustering)
        else:
            score, _ = self.sampler.sample(logger, self.model, self.prior, train_loader) #valid*unique
                                                                                            #mean(degress,orbit,clustering)
        return score