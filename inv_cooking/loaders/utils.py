import torch
from torch.utils.data.sampler import Sampler


class RandomSamplerWithStateIterator:
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed
        g = torch.Generator()
        g.manual_seed(seed)
        self.indices = torch.randperm(len(self.data_source), generator=g).tolist()
        self.epoch = 0
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.data_source):
            raise StopIteration
        item = self.indices[self.index]
        self.index += 1
        return item

    def set_epoch(self, epoch):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        self.epoch = epoch
        self.indices = torch.randperm(len(self.data_source), generator=g).tolist()
        self.index = 0

    def increase_epoch(self):
        self.set_epoch(self.epoch + 1)


class RandomSamplerWithState(Sampler):
    def __init__(self, data_source, batch_size, seed):
        self.iterator = RandomSamplerWithStateIterator(data_source, seed)
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.iterator.data_source)

    def set_state(self, epoch, step):
        self.iterator.set_epoch(epoch)
        self.iterator.index = step * self.batch_size


def increase_loader_epoch(data_loader):
    data_loader.sampler.iterator.increase_epoch()
