import jittor
from jittor.dataset import DataLoader, Sampler
from jittor.dataset import Dataset


# TODO:
#  1. using dataloader load 3dpw  and print shape

class RandomSampler(Sampler):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size'] * checkpoint['batch_idx']:]
        else:
            n = len(self.data_source)
            self.dataset_perm = jittor.misc.randperm(n)
            self.dataset_perm = jittor.misc.tolist(self.dataset_perm)
            self.perm = jittor.misc.randperm(n)
            self.perm = jittor.misc.tolist(self.perm)

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return len(self.perm)


class SequentialSampler(Sampler):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size'] * checkpoint['batch_idx']:]
        else:
            self.dataset_perm = list(range(len(self.data_source)))
            self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return len(self.perm)


def CheckpointDataLoader(dataset: Dataset, checkpoint=None, batch_size=32,
                         shuffle=False, num_worker=0, drop_last=True, num_workers=8):
    # if shuffle:
    #     sampler = RandomSampler(dataset, checkpoint)
    # else:
    #     sampler = SequentialSampler(dataset, checkpoint)
    if checkpoint is not None:
        dataset.checkpoint_batch_idx = checkpoint['batch_idx']
    else:
        dataset.checkpoint_batch_idx = 0
    if shuffle:
        print(">>>>>>>>>>using shuffle data")
    else:
        print(">>>>>>>>>>NOT using shuffle data")
    dataloader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader
