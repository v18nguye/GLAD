import pdb
import torch
from torch_geometric.data import InMemoryDataset
from torch.utils.data import TensorDataset
from ._process import process_generic_data, process_molecule_data

class GraphDataset(InMemoryDataset):
    def __init__(self, datset, root, train=False, transform=None):
        self.train = train
        self.datset = datset
        root = root + self.datset
        super().__init__(root, transform=transform)
        self.data = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        if self.train:
            return ['train_data.pt']
        else:
            return ['test_data.pt']
    
    def process(self):
        # Read data into huge `Data` list.
        if self.datset in ['community', 'ego', 'ENZYMES']:
            train_list, test_list = process_generic_data(self.datset)
        elif self.datset in ['ZINC250k', 'QM9']:
            train_list, test_list = process_molecule_data(self.datset)
        else:
            raise ValueError('Dataset Invalid: '+self.datset)
        
        if self.train:
            x_tensor_feat = torch.cat([i[0].unsqueeze(0) for i in train_list], dim=0)
            adj_tensor_feat = torch.cat([i[1].unsqueeze(0) for i in train_list], dim=0)
        else:
            x_tensor_feat = torch.cat([i[0].unsqueeze(0) for i in test_list], dim=0)
            adj_tensor_feat = torch.cat([i[1].unsqueeze(0) for i in test_list], dim=0)
            
        torch.save(TensorDataset(x_tensor_feat, adj_tensor_feat), self.processed_paths[0])