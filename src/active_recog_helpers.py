import numpy as np
from collections import defaultdict

def diverse_permutation(labels):
    """
    Returns a permutation of indices such that label diversity is maximized early on.
    
    Args:
        labels (list or np.array): A list of labels, e.g., [0, 1, 2, 0, 0, 2]
    
    Returns:
        list: A permutation of indices that prioritizes label diversity early.
    """
    labels = np.array(labels)
    unique_labels = np.random.permutation(np.unique(labels))
    label_indices = defaultdict(list)
    
    # Collect indices for each label
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    # Shuffle indices within each label
    for label in label_indices:
        np.random.shuffle(label_indices[label])
    
    # Interleave labels to maximize early diversity
    permuted_indices = []
    
    while any(label_indices.values()):
        for label in unique_labels:
            if label_indices[label]:
                permuted_indices.append(label_indices[label].pop(0))
    
    return permuted_indices

## Example usage:
#labels = [0, 1, 2, 0, 0, 2, 0, 0, 0, 0, 2, 1]
#permuted_indices = diverse_permutation(labels)
#permuted_labels = [labels[i] for i in permuted_indices]
#print("Permuted Indices:", permuted_indices)
#print("Permuted Labels:", permuted_labels)