import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class BridgeLoss(nn.Module):
    def __init__(self):
        super(BridgeLoss, self).__init__()
        
    def forward(self, gt, pred):
        '''
        @params
            gt: {R}^(bs, n, L)
            pred: {R}^(bs, n, L)
        '''
        loss = torch.square((gt - pred))
        loss = loss.reshape(gt.shape[0], -1)
        loss = torch.mean(loss, dim=-1)
        loss = torch.mean(loss)
        
        return loss

class FSQLoss(nn.Module):
    def __init__(self, scale):
        self.scale = scale
        super(FSQLoss, self).__init__()
    
    def forward(self, x_gt, e_gt, x_prd, e_prd, weight= 5.):
        '''
        @params
            x_gt: {0, 1}^(bs, n, num_cls_x)
            x_prd: {R}^(bs, n, num_cls_x)
            
            e_gt: {0, ..., num_cls_e}^(bs, n, n)
            e_prd: [0,1)^(bs, n, n)
        '''
        loss_x = torch.square((x_gt - x_prd))
        loss_x = loss_x.reshape(loss_x.shape[0], -1)
        loss_x = torch.mean(loss_x, dim=-1)
        
        e_gt = e_gt / self.scale
        loss_e = torch.square((e_gt - e_prd))
        loss_e = loss_e.reshape(e_gt.shape[0], -1)
        loss_e = torch.mean(loss_e, dim=-1)
        loss = torch.mean(loss_x) + weight*torch.mean(loss_e)
        
        # x_acc
        _x_prd = x_prd.detach()
        _x_prd = _x_prd.argmax(-1)
        _x_gt = x_gt.argmax(-1)
        _size_x = _x_gt.shape[0]* _x_gt.shape[1]
        x_acc = ((_x_gt == _x_prd).sum())/_size_x
        
        # e_acc
        _e_prd = e_prd.detach()
        _e_gt = (e_gt*self.scale).round().long()
        _e_prd = (_e_prd*self.scale).round().long()
        _size_e = _e_gt.shape[0]* _e_gt.shape[1]* _e_gt.shape[2]
        e_acc = ((_e_gt == _e_prd).sum())/_size_e
        return loss, e_acc, x_acc