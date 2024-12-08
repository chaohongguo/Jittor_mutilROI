import jittor
import numpy as np
from .base_dataset import BaseDataset


class MixedDataset(jittor.dataset.Dataset):
    def __init__(self, options, **kwargs):
        super().__init__()
        # TODO lsp-orig ???
        self.dataset_list = ['h36m', "coco", "3dpw", "lspet", "mpii", "mpi-inf-3dhp"]
        self.dataset_dict = {'h36m': 0, "coco": 1, "3dpw": 2, "lspet": 3, "mpii": 4, "mpi-inf-3dhp": 5}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        total_length = sum([len(ds) for ds in self.datasets])
        print(f'Total length:{total_length}')
        print(f'Dataset:{ds1.dataset},Length{ds1}' for ds1 in self.datasets)
        # length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        self.partition = [0.9, 0, 0.1, 0, 0]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()

        for i in range(len(self.dataset_list)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
