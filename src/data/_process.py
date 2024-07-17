
import os
import pdb
import json
import torch
import pickle
import numpy as np
import networkx as nx
import torch.nn.functional as F

DATA_DIR = os.environ['DATA_DIR']

def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a: (x, a), result[0], result[1]))

def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a

def get_transform_fn(dataset):
    if dataset == 'QM9':
        def transform(g):
            # 6: C, 7: N, 8: O, 9: F
            # x: (N), adj: (4, N, N) no AROMATIC bond
            x, adj = g
            
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)
            adj = adj.argmax(axis=0)
            adj =  torch.tensor(adj)
            # 0:vitrual-edge, 1:S, 2:D, 3:T
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            mask_edge = adj.sum(-1) > 0
            
            mask_atom =  x > 0
            num_atom = sum(mask_atom)
            
            # only consider molecule having more than one atom,
            # and have at least one bond type.
            if num_atom > 1 and (mask_atom == mask_edge.numpy()).all():
                x_ = np.zeros((9, 4)) # considering no vitrual node.

                mask = x >= 6
                indices = x[mask] - 6
                x_[mask, indices] = 1

                x = torch.tensor(x_).to(torch.float32)
                adj = adj.to(torch.float32)
                # one-hot encoding x
                data = (x, adj)
            else:
                data = None
            return data
        
    elif dataset == 'ZINC250k':
        def transform(g):
            x, adj = g
            
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)
            adj = adj.argmax(axis=0)
            adj = torch.tensor(adj)
            
            # 0:vitrual-edge, 1:S, 2:D, 3:T
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            mask_edge = adj.sum(-1) > 0
            
            mask_atom =  x > 0
            num_atom = sum(mask_atom)
            
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53] # considering no virtual node.
            if num_atom > 1 and (mask_atom == mask_edge.numpy()).all():
                x_ = np.zeros((38, 9), dtype=np.float32)
                for i in range(38):
                    # no virtual node consideration
                    if x[i] in zinc250k_atomic_num_list:
                        ind = zinc250k_atomic_num_list.index(x[i])
                        x_[i, ind] = 1.
                # one-hot encoding x
                x = torch.tensor(x_).to(torch.float32)
                adj = adj.to(torch.float32)
                data = (x, adj)
            else:
                data = None
            return data        
    return transform

def process_generic_data(dataset):
    _num_xfeat = {'community': 10 - 1, 'ego': 17 - 1, 'ENZYMES': 10 - 1} # no taking account to zero degree.
    _max_node_num = {'community': 20, 'ego': 18, 'ENZYMES': 125}
    
    if dataset in ['community', 'ego']:
        file_path = os.path.join(DATA_DIR, dataset+'_small.pkl')
    elif dataset in ['ENZYMES']:
        file_path = os.path.join(DATA_DIR, dataset+'.pkl')
    else:
        raise ValueError('Dataset Invalid: ',dataset)
    
    with open(file_path, 'rb') as f:
        raw_list = pickle.load(f)
        
    processed_list = []
    for g in raw_list:
        assert isinstance(g, nx.Graph)
        
        node_list = []
        for v, _ in g.nodes.data('feature'):
            node_list.append(v)
            
        adj = nx.to_numpy_matrix(g, nodelist=node_list)
        pad_adj = pad_adjs(adj, node_number=_max_node_num[dataset])
        pad_adj = torch.tensor(pad_adj).to(torch.long)
        
        x_feat = pad_adj.sum(dim=-1).to(torch.long)
        mask = (x_feat.gt(0.1)*1.0).unsqueeze(-1)
        x_feat[mask.gt(0.1).squeeze()] = x_feat[mask.gt(0.1).squeeze()] - 1 # no taking account to zero degree.
        # one hot encoding
        x_feat_ = F.one_hot(x_feat, num_classes=_num_xfeat[dataset]).to(torch.float32) # degree feature
        # filter out virtual node attr
        x_feat =  x_feat_ * mask
        
        data = (x_feat, pad_adj)
        processed_list.append(data)

    test_len = int(len(processed_list)*0.2)
    train_list, test_list = processed_list[test_len:], processed_list[:test_len]
        
    return train_list, test_list  
    
def process_molecule_data(dataset):
    mols = load_mol(os.path.join(DATA_DIR, f'{dataset.lower()}_kekulized.npz'))
    
    with open(os.path.join(DATA_DIR, f'valid_idx_{dataset.lower()}.json')) as f:
        test_idx = json.load(f)
            
    if dataset == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
        
    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')
    
    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]
    
    train_list = []
    test_list = []
    
    transform = get_transform_fn(dataset)
    
    for g in train_mols:
        data = transform(g)
        if data is not None:
            train_list.append(data)
            
    for g in test_mols:
        data = transform(g)
        if data is not None:
            test_list.append(data)
    print(f'Number of processed training mols: {len(train_list)} | Number of processed test mols: {len(test_list)}')
    
    return train_list, test_list