import numpy as np

class SequentialSampler:
    def __init__(self,num_instances):
        self.num_instances = num_instances

    def __iter__(self):
        indices = np.arange(self.num_instances)
        yield from indices

    def __len__(self):
        return self.num_instances

class RandomSampler:
    def __init__(self,num_instances):
        self.num_instances = num_instances

    def __iter__(self):
        indices = np.random.permutation(np.arange(self.num_instances))
        yield from indices

    def __len__(self):
        return self.num_instances

class BatchSampler:

    def __init__(self, batch_size, sampler, drop_last):
        self.batch_size = batch_size
        self.sampler = sampler
        self.drop_last = drop_last

    def __iter__(self):
        batch_indices = []
        for idx in self.sampler:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if len(batch_indices) > 0 and not self.drop_last:
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore