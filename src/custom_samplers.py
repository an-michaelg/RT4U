import itertools
from collections import defaultdict
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)

import numpy as np
import torch
from torch.utils.data import Sampler

# class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
    
    # def __init__(self, data: List[str], batch_size: int) -> None:
        # self.data = data
        # self.batch_size = batch_size
    
    # def __len__(self) -> int:
        # return (len(self.data) + self.batch_size - 1) // self.batch_size
        
    # def __iter__(self) -> Iterator[List[int]]:
        # sizes = torch.tensor([len(x) for x in self.data])
        # for batch in torch.chunk(torch.argsort(sizes), len(self)):
            # yield batch.tolist()
            
            
# class WeightedRandomSampler(Sampler[int]):
    # r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    # Args:
        # weights (sequence)   : a sequence of weights, not necessary summing up to one
        # num_samples (int): number of samples to draw
        # replacement (bool): if ``True``, samples are drawn with replacement.
            # If not, they are drawn without replacement, which means that when a
            # sample index is drawn for a row, it cannot be drawn again for that row.
        # generator (Generator): Generator used in sampling.

    # Example:
        # >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        # >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        # [4, 4, 1, 4, 5]
        # >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        # [0, 1, 4, 3, 2]
    # """

    # weights: torch.Tensor
    # num_samples: int
    # replacement: bool

    # def __init__(
        # self,
        # weights: Sequence[float],
        # num_samples: int,
        # replacement: bool = True,
        # generator=None,
    # ) -> None:
        # if (
            # not isinstance(num_samples, int)
            # or isinstance(num_samples, bool)
            # or num_samples <= 0
        # ):
            # raise ValueError(
                # f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            # )
        # if not isinstance(replacement, bool):
            # raise ValueError(
                # f"replacement should be a boolean value, but got replacement={replacement}"
            # )

        # weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        # if len(weights_tensor.shape) != 1:
            # raise ValueError(
                # "weights should be a 1d sequence but given "
                # f"weights have shape {tuple(weights_tensor.shape)}"
            # )

        # self.weights = weights_tensor
        # self.num_samples = num_samples
        # self.replacement = replacement
        # self.generator = generator

    # def __iter__(self) -> Iterator[int]:
        # rand_tensor = torch.multinomial(
            # self.weights, self.num_samples, self.replacement, generator=self.generator
        # )
        # yield from iter(rand_tensor.tolist())

    # def __len__(self) -> int:
        # return self.num_samples   
        

# balanced sampling wo replacement BUT each batch shares a label class
# n_batches = sum_c( ceil(N_of_class_c / batch_size) )
def sample_classbags_wo_replacement(arr, batch_size=4):
    
    # first separate the arr into different classes
    indices_of_class = defaultdict(list)
    batches = []
    for i, c in enumerate(arr):
        indices_of_class[c].append(i)
    
    # for each class, shuffle the indices and put them into groups of batch_size
    for c, inds in indices_of_class.items():
        newly_ordered = np.random.permutation(inds)
        n_sections = int(np.ceil(len(newly_ordered) / batch_size))
        batches.extend(np.array_split(newly_ordered, n_sections))
        
    batches_list = []
    for k in np.random.permutation(len(batches)):
        batches_list.append(batches[k])
    return batches_list
    
    
# label-balanced sampling, each batch shares a label class
# n_batches = ceil(N / batch_size)
def sample_classbags_balance(arr, batch_size=4):
    N = len(arr)
    
    # first separate the arr into different classes
    indices_of_class = defaultdict(list)
    counts = []
    for i, c in enumerate(arr):
        indices_of_class[c].append(i)
    
    # determine the odds of sampling each class
    for c, inds in indices_of_class.items():
        counts.append(len(inds))
    
    odds = 1 / np.array(counts)
    odds = odds / np.sum(odds) # norm to 1
    #print(odds)
    
    # sample a class, then from the class, sample batch_size indices at random
    n_batches = int(np.ceil(N / batch_size))
    n_classes = len(odds)
    sample_of_classes = np.random.choice(n_classes, n_batches, p=odds)
    batches_list = []
    for c in sample_of_classes:
        batches_list.append(np.random.choice(indices_of_class[c], batch_size))
        
    return batches_list
    
    
# balanced sampling wo replacement, each batch represents a single study
# each study is forced to have batch_size images, so some images may be repeated/excluded at random
# also, we are assuming each study has 1 label, which should be reasonable
# n_batches = n_studies
def sample_studies_wo_replacement(study_id, batch_size=4):

    # first group the samples by study
    indices_of_study = defaultdict(list)
    for i, s in enumerate(study_id):
        indices_of_study[s].append(i)
        
    # sample batch_size images from each study
    batches = []
    for s, inds in indices_of_study.items():
        sample_of_images_from_s = np.random.choice(inds, batch_size)
        batches.append(sample_of_images_from_s)
        
    batches_list = []
    for k in np.random.permutation(len(batches)):
        batches_list.append(batches[k])
        
    return batches_list
    

# n_batches = n_studies
def sample_studies_balance(study_id, arr, batch_size=4):
    
    # group the samples by study
    indices_of_study = defaultdict(list)
    for i, s in enumerate(study_id):
        indices_of_study[s].append(i)
        
    # count the studies by class
    n_classes = len(np.unique(arr))
    counts = np.zeros(n_classes)
    s_ids, s_classes = [], []
    for s, inds in indices_of_study.items():
        s_ids.append(s)
        s_class = arr[inds[0]]
        s_classes.append(s_class)
        counts[s_class] += 1
    
    # determine the odds of sampling each class
    class_odds = 1 / counts
    s_odds = class_odds[s_classes]
    s_odds = s_odds / np.sum(s_odds) # norm to 1
    
    # sample each study according to study_odds to yield a list of studies
    sample_of_studies = np.random.choice(s_ids, len(s_ids), p=s_odds)
    batches_list = []
    for s in sample_of_studies:
        sample_of_images_from_s = np.random.choice(indices_of_study[s], batch_size)
        batches_list.append(sample_of_images_from_s)
        
    return batches_list

class MILSampler(Sampler[List[int]]):
    
    def __init__(self, study_id, as_label, batch_size=4, groupby_study=False, balance_label=False) -> None:
        self.batch_size = batch_size
        self.groupby_study = groupby_study
        self.balance_label = balance_label
        
        self.study_id = study_id
        self.as_label = as_label
        N = len(self.as_label)
        
        # compute the sampling weights based on settings
                
        if self.groupby_study == False:
            if self.balance_label == False:
                # balanced sampling wo replacement BUT each batch shares a label class
                batches = sample_classbags_wo_replacement(self.as_label, self.batch_size)
                
            else:
                # label-balanced sampling, each batch shares a label class
                batches = sample_classbags_balance(self.as_label, self.batch_size)
        else:
            if self.balance_label == False:
                # balanced sampling wo replacement, each batch represents a single study
                batches = sample_studies_wo_replacement(self.study_id, self.batch_size)
            else:
                # label-balanced sampling, each batch represents a single study
                batches = sample_studies_balance(self.study_id, self.as_label, self.batch_size)

        self.n_batches = len(batches)
        
    def __iter__(self) -> Iterator[int]:
        if self.groupby_study == False:
            if self.balance_label == False:
                # balanced sampling wo replacement BUT each batch shares a label class
                batches = sample_classbags_wo_replacement(self.as_label, self.batch_size)
                
            else:
                # label-balanced sampling, each batch shares a label class
                batches = sample_classbags_balance(self.as_label, self.batch_size)
        else:
            if self.balance_label == False:
                # balanced sampling wo replacement, each batch represents a single study
                batches = sample_studies_wo_replacement(self.study_id, self.batch_size)
            else:
                # label-balanced sampling, each batch represents a single study
                batches = sample_studies_balance(self.study_id, self.as_label, self.batch_size)
        yield from iter(batches)

    def __len__(self) -> int:
        return self.n_batches  
                

if __name__ == '__main__':
    batch_size = 2
    arr = np.array([0, 1, 1, 0, 0, 1, 0, 0])
    study = np.array([0, 1, 1, 2, 2, 3, 4, 4])
    #print(sample_classbags_balance(arr, batch_size))
    #print(sample_studies_balance(study, arr, batch_size))
    sampler = MILSampler(study, arr, batch_size, groupby_study=False, balance_label=True)
    for i, j in enumerate(sampler):
        print(f'{i}, {j}')