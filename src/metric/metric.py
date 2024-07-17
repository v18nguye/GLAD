from ._base import SpectreSamplingMetric, MolSamplingMetric

class GenericMetric(SpectreSamplingMetric):
    def __init__(self, train_loader, test_loader):
        super(GenericMetric, self).__init__(train_loader, test_loader,
                         compute_emd=True,
                         metrics_list=['degree', 'clustering', 'orbit'])
        
class Qm9Metric(MolSamplingMetric):
    def __init__(self, data_dir=None):
        super().__init__(dataset='QM9', data_dir=data_dir)

class Zinc250Metric(MolSamplingMetric):
    def __init__(self, data_dir=None):
        super().__init__(dataset='ZINC250k', data_dir=data_dir)