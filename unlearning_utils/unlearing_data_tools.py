import numpy as np
import copy
from torch.utils.data import Dataset

def get_idx_by_unlearn_class(labels, unlearn_class) -> np.ndarray:
    """
    instances (list-like):
    labels (list-like):
    classes (list-like): unique classes
    unlearn_class (list-like): unlearn classes
    """
    labels = np.array(labels)
    all_idxs = []
    for c in unlearn_class:
        idxs = np.where(labels == c)[0]
        all_idxs.append(idxs)
    all_idxs = np.concatenate(all_idxs)
    return all_idxs

def get_idx_by_unlearn_noise(labels, unlearn_class: list, noise_rate):
    """
    labels (list-like):
    unlearn_class (list-like): unlearn classes
    noise_rate (float, list-like): noise rate
    """
    labels = np.array(labels)
    all_idxs = []
    if not isinstance(noise_rate, float):
        assert len(unlearn_class) == len(noise_rate)
    
    for i, c in enumerate(unlearn_class):
        idxs = np.where(labels == c)[0]
        if isinstance(noise_rate, float):
            noise_num = int(len(idxs) * noise_rate)
        else:
            noise_num = int(len(idxs) * noise_rate[i])
        noise_idxs = np.random.choice(idxs, noise_num, replace=False)
        all_idxs.append(noise_idxs)
    all_idxs = np.concatenate(all_idxs)
    labels[all_idxs] += 1
    return all_idxs, labels


def get_subset(dataset, idxs) -> Dataset:
    subset = copy.deepcopy(dataset)
    if isinstance(subset.targets, list):
        subset.targets = np.array(subset.targets)
    subset.targets = subset.targets[idxs]
    subset.data = subset.data[idxs]
    return subset