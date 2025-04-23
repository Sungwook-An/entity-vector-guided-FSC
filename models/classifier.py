import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingNetworkClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.metric = args.metric  # "cosine" or "euclidean"
        self.temperature = args.classifier_temperature if hasattr(args, 'classifier_temperature') else 1.0

    def forward(self, query_features, support_features, support_labels):
        """
        Args:
            query_features:   [N*Q, D]
            support_features: [N*K, D]
            support_labels:   [N*K]

        Returns:
            logits: [N*Q, N]
        """
        if query_features.dim() == 4:
            query_features = query_features.mean(dim=[2, 3])
        
        if support_features.dim() == 3:
            N_K, H_W, D = support_features.shape
            support_features = support_features.mean(dim=1)
        else:
            D = support_features.shape[1]
            
        if support_labels.dim() == 2:
            support_labels = support_labels.view(-1)
            
        NQ = query_features.size(0)
        NK = support_features.size(0)

        # Normalize features for cosine similarity
        if self.metric == "cosine":
            query_norm = F.normalize(query_features, dim=-1)  # [N*Q, D]
            support_norm = F.normalize(support_features, dim=-1)  # [N*K, D]
            sim_matrix = torch.matmul(query_norm, support_norm.T)  # [N*Q, N*K]
        elif self.metric == "euclidean":
            query_sq = query_features.pow(2).sum(dim=1, keepdim=True)  # [N*Q, 1]
            support_sq = support_features.pow(2).sum(dim=1).unsqueeze(0)  # [1, N*K]
            sim_matrix = - (query_sq + support_sq - 2 * torch.matmul(query_features, support_features.T))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # num_classes = support_labels.max().item() + 1
        num_classes = support_labels.max().item() + 1
        one_hot_labels = F.one_hot(support_labels, num_classes=num_classes).float()  # [N*K, N]

        # Weighted sum of similarity scores to each class
        logits = torch.matmul(sim_matrix, one_hot_labels)  # [N*Q, N]
        logits = logits / self.temperature

        return logits
