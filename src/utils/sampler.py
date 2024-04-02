import pdb
import torch
from ._base import SamplerBase
from ..metric.metric import GenericMetric, Qm9Metric, Zinc250Metric
from ..model._lib import build_mask

METRIC = {'QM9': Qm9Metric,
               'ZINC250k': Zinc250Metric,
               'Generic': GenericMetric}

class MolSampler(SamplerBase):
    def __init__(self, n, _dataset):
        super().__init__(n)
        self.metric_func = METRIC[_dataset]()
        
    def sample(self, logger, model, prior, train_loader):
        '''
        @params
            train_loader:
                sample # sample from training
        '''
        e_out_list = []
        x_out_list = []
        p_device = next(prior.model.parameters()).device
        m_device = next(model.parameters()).device
        
        for _ in range(10000):
            
            for batch in train_loader:
                # randomly select masking samples from training data
                if torch.randn(1) > 0.5:
                    e_in = batch[1].to(p_device)
                    mask = build_mask(e_in)

                    with torch.no_grad():
                        z0 = prior.sample(mask.shape[0], e_in.device, mask)
                        
                        z0 = z0.detach().cpu().to(m_device)
                        mask = mask.detach().cpu().to(m_device)
                        
                        z0 = model.quantizer.map_to_dec(z0, mask)
                        y_dec = z0.mean(1)
        
                        adj_dec = torch.bmm(z0, z0.transpose(1,2)).sigmoid()
                        adj_dec = adj_dec * mask.unsqueeze(1) * mask.unsqueeze(2)
                        adj_dec = 1/2 * (adj_dec + adj_dec.transpose(1,2))
                        adj_dec = adj_dec * (~torch.eye(adj_dec.shape[-1], device=adj_dec.device).gt(0).unsqueeze(0)).float()
        
                        x_out, e_out = model.dec(z0, adj_dec, y_dec, mask)
                        x_out, e_out = x_out.cpu(), e_out.cpu()

                        # scale back bond values
                        e_out =  (e_out*model.dec.scale).round()
                        
                        # assign virtual node with mask
                        num_cls = x_out.shape[-1]
                        x_out = x_out.argmax(-1)
                        x_out[~mask.gt(0.1)] = num_cls
                        x_out_list.append(x_out)
                        e_out_list.append(e_out)
                        
                    logger.info(f'sampling: {sum([x.shape[0] for x in x_out_list])}')
                
                if sum([x.shape[0] for x in x_out_list]) >= self.n:
                    break
            
            if sum([x.shape[0] for x in x_out_list]) >= self.n:
                break
        x_out_tensor = torch.cat(x_out_list, dim=0)[:self.n,...]
        e_out_tensor = torch.cat(e_out_list, dim=0)[:self.n,...]
        nvun, result = self.metric_func(x_out_tensor, e_out_tensor, logger)
        return nvun, result
    
class SpecSampler(SamplerBase):
    def __init__(self, n, train_loader, test_loader):
        super().__init__(n)
        self.metric_func = METRIC['Generic'](train_loader, test_loader)
        
    def sample(self, logger, model, prior, train_loader):
        e_out_list = []
        
        p_device = next(prior.model.parameters()).device
        m_device = next(model.parameters()).device
        
        for _ in range(10000):
            for batch in train_loader:
                # randomly select masking samples from training data
                if torch.randn(1) > 0.5:
                    e_in = batch[1].to(p_device)
                    mask = build_mask(e_in)
                    logger.info(f'start sampling ...')
                    with torch.no_grad():
                        z0 = prior.sample(mask.shape[0], e_in.device, mask)
                        
                        z0 = z0.detach().cpu().to(m_device)
                        mask = mask.detach().cpu().to(m_device)
                        
                        z0 = model.quantizer.map_to_dec(z0, mask)
                        y_dec = z0.mean(1)

                        adj_dec = torch.bmm(z0, z0.transpose(1,2)).sigmoid()
                        adj_dec = adj_dec * mask.unsqueeze(1) * mask.unsqueeze(2)
                        adj_dec = 1/2 * (adj_dec + adj_dec.transpose(1,2))
                        adj_dec = adj_dec * (~torch.eye(adj_dec.shape[-1], device=adj_dec.device).gt(0.1).unsqueeze(0)).float()
                        
                        _, e_out = model.dec(z0, adj_dec, y_dec, mask)
                        e_out = e_out.cpu()
                        # scale back bond values
                        e_out = (e_out*model.dec.scale).round()
                        e_out_list.append(e_out)
                        
                    logger.info(f'sampling: {sum([x.shape[0] for x in e_out_list])}')
                
                if sum([x.shape[0] for x in e_out_list]) >= self.n:
                    break
                
            if sum([x.shape[0] for x in e_out_list]) >= self.n:
                    break
        e_out_tensor = torch.cat(e_out_list, dim=0)[:self.n,...]
        e_out_sample = [e_out_tensor[i] for i in range(e_out_tensor.shape[0])]
        res = self.metric_func(e_out_sample)
        logger.info(f'result: {res}')
        score = - (res['degree'] + res['clustering'] + res['orbit'])/3
        return score, res