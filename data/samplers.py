import torch
import numpy as np


class CategoriesSampler:
    """
    CategoriesSampler for N-way K-shot learning.
    For each episode:
        - Randomly sample N classes
        - For each class, sample K + Q examples (support + query)
    """
    
    def __init__(self, labels, n_batch, n_way, n_shot, n_query):
        """
        Args:
            labels (list): List of labels for each example.
            n_batch (int): Number of batches.
            n_way (int): Number of classes per episode.
            n_shot (int): Number of examples per class in the support set.
            n_query (int): Number of examples per class in the query set.
        """
        
        self.labels = np.array(labels)
        self.n_batch = n_batch
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        
        self.class_indices = []
        for i in range(max(labels) + 1):
            indices = np.argwhere(self.labels == i).reshape(-1)
            self.class_indices.append(torch.from_numpy(indices))
            
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            # Randomly sample N classes
            classes = torch.randperm(len(self.class_indices))[:self.n_way]
            
            support_indices = []
            query_indices = []
            
            for c in classes:
                indices = self.class_indices[c]
                perm = torch.randperm(len(indices))
                selected = indices[perm[:self.n_shot + self.n_query]]
                support_indices.append(selected[:self.n_shot])
                query_indices.append(selected[self.n_shot:])
            
            support_indices = torch.stack(support_indices).view(-1)
            query_indices = torch.stack(query_indices).view(-1)
            
            yield support_indices, query_indices
        