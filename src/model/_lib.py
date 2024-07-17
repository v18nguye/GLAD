import math
import torch

def assert_correctly_masked( z, mask):
    assert (z * (1 - mask.long())).abs().max().item() < 1e-4, 'Variables not masked properly.'

def init_glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def init_zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def build_mask(m, eps=0.1):
    if m.shape.__len__() == 3:
        # masking with edge matrix
        mask = torch.abs(m).sum(-1).gt(eps).to(dtype=torch.float32)
    else:
        raise ValueError('masking size: ', m.shape, ' invalid!')
    return mask

# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):
    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs

# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        deg_inv_sqrt = x.abs().sum(dim=-1).clamp(min=1).pow(-0.5)
        x_normalized = deg_inv_sqrt.unsqueeze(-1) * x * deg_inv_sqrt.unsqueeze(-2)
        x_ = torch.bmm(x_, x_normalized)
        mask = torch.ones([x.shape[-1], x.shape[-1]]) - torch.eye(x.shape[-1])
        x_ = x_ * mask.unsqueeze(0).to(x.device)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)
    return xc