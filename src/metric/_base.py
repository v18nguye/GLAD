import torch.nn as nn
# from moses.metrics.metrics import get_all_metrics
from .molsets import get_all_metrics
from .mol_utils import *
from .spec_utils import  degree_stats, clustering_stats, orbit_stats_all

class MolSamplingMetric(nn.Module):
    def __init__(self, dataset, data_dir):
        super(MolSamplingMetric, self).__init__()
        self.dataset = dataset
        if data_dir is None:
            self.data_dir = '/home/users/n/nguyeva2/dev/code/GDSS/data'
        else:
            self.data_dir = data_dir
        self.train_smiles, self.test_smiles, self.test_graph_list = self._load_data()
        
    def _load_data(self):
        '''load canonicalized smiles
        '''
        train, test = load_smiles(self.data_dir, self.dataset)
        train = canonicalize_smiles(train)
        test = canonicalize_smiles(test)
        with open(f'{self.data_dir}/{self.dataset.lower()}_test_nx.pkl', 'rb') as f:
            test_graph_list = pickle.load(f) # for NSPDK MMD, smiles -> nx, no kekulization, no canoncalization.
        return train, test, test_graph_list
    
    def forward(self, x_gen, adj_gen, logger):
        '''
        @params
            x_gen: {R}^(B, N) 
                (..., d_x - 1: vitrual)
                generated class probabilities of each class.
            
            adj_gen: {0,1,2,3}^(B, N, N) 
                (0: vitrual, 1:S, 2:D, 3:T)
                
            # x_gen: {R}^(B, N, d_x) 
                (..., d_x - 1: vitrual)
                generated class probabilities of each class.
                
            # adj_gen: {R}^(B, N, N, 4) 
                (0: vitrual, 1:S, 2:D, 3:T)
                generated class probabilities of each class.
        '''
        
        gen_mols, num_mols_wo_correction = gen_mol(x_gen, adj_gen, self.dataset)
        num_mols = len(gen_mols)
        
        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]
        
        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device='cuda', n_jobs=4, test=self.test_smiles, train=self.train_smiles)
        scores_nspdk = nspdk_stats(self.test_graph_list, mols_to_nx(gen_mols))
        
        result = {}
        logger.info(f'Number of molecules: {num_mols}')
        logger.info(f'validity w/o correction: {num_mols_wo_correction / num_mols}')
        result['valid_wo_correct'] = num_mols_wo_correction / num_mols
        for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']:
            logger.info(f'{metric}: {scores[metric]}')
            result[metric] = scores[metric]
        logger.info(f'NSPDK MMD: {scores_nspdk}')
        result['nspdk_mmd'] = scores_nspdk
        
        nvun = (num_mols_wo_correction / num_mols)*scores[f'unique@{len(gen_smiles)}']
        
        return nvun, result


class SpectreSamplingMetric(nn.Module):
    def __init__(self, train_loader, test_loader, compute_emd, metrics_list):
        super(SpectreSamplingMetric, self).__init__()
        self.train_nx_graph = self.loader_to_nx(train_loader)
        self.test_nx_graph = self.loader_to_nx(test_loader)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list

    def loader_to_nx(self, loader):
        '''
        @params
            loader: torch-dataloader({0, 1, ..., d_x -1}^(B, N),
                                            {0, 1}^(B, N, N)).
        '''
        nx_graph_list = []
        for _, adj in loader:
            data_list = [(adj[i,:,:], adj.sum(-1).gt(0.5).sum()) for i in range(adj.shape[0])]
            for a_i, num_node in data_list:
                a_i = a_i[:num_node,:num_node].type(torch.long)
                nx_graph = nx.from_numpy_array(a_i.numpy().astype(bool))
                nx_graph_list.append(nx_graph)
        return nx_graph_list
    
    def torch_to_nx(self, generated_graphs):
        '''converting torch to nx
        @params
            generated_graphs: list({0,1}^(N, N))

        '''
        nx_graph_list = []
        for adj in generated_graphs:
            A = adj.bool().cpu().numpy()
            nx_graph = nx.from_numpy_array(A)
            nx_graph_list.append(nx_graph)
        return nx_graph_list

    def forward(self, generated_graphs):
        '''
        @params
            generated_graphs: list({0,1}^(N, N))
                generated graphs from model
        '''
        print(f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(self.test_nx_graph)}"
              f" test graphs -- emd computation: {self.compute_emd}")
        print("Building networkx for generated graphs...")
        gen_nx_graph = self.torch_to_nx(generated_graphs)
            
        to_log = {}
        if 'degree' in self.metrics_list:
            print("Computing degree stats..")
            degree = degree_stats(self.test_nx_graph, gen_nx_graph, is_parallel=True,
                                  compute_emd=self.compute_emd)
            to_log['degree'] =  degree

        if 'clustering' in self.metrics_list:
            print("Computing clustering stats...")
            clustering = clustering_stats(self.test_nx_graph, gen_nx_graph, bins=100, is_parallel=True,
                                          compute_emd=self.compute_emd)
            to_log['clustering'] = clustering

        if 'orbit' in self.metrics_list:
            print("Computing orbit stats...")
            orbit = orbit_stats_all(self.test_nx_graph, gen_nx_graph, compute_emd=self.compute_emd)
            to_log['orbit'] = orbit

        return to_log