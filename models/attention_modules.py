import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseAttention(nn.Module):
    def __init__(self, image_dim, text_dim, reduction_ratio=16):
        super(ChannelWiseAttention, self).__init__()
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, image_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(image_dim, image_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(image_dim // reduction_ratio, image_dim),
            nn.Sigmoid()
        )
        
    def forward(self, image_feat, text_feat):
        """
        Args:
            image_feat (Tensor): Shape (B, C, H, W)
            text_feat (Tensor): Shape (B, D)
            
        Returns:
            Tensor of shape (B, C, H, W): Attention-modulated image features
        """
        B, C, H, W = image_feat.shape
        
        image_feat_proj = self.image_proj(image_feat)  # (B, C, H, W)
        pooled_feat = F.adaptive_avg_pool2d(image_feat_proj, 1).view(B, C)
        
        text_proj = self.text_proj(text_feat)  # (B, C)
        fusion = pooled_feat + text_proj # (B, C)
        
        weights = self.fc(fusion).view(B, C, 1, 1)  # (B, C, 1, 1)
        out = image_feat * weights
        
        return out
    

class CrossAttention(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim=512):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        self.key_proj = nn.Conv2d(image_dim, hidden_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(image_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, image_dim, kernel_size=1)

    def forward(self, image_feat, text_feat):
        """
        Args:
            image_feat: Tensor of shape (B, C, H, W)
            text_feat: Tensor of shape (B, D)
        Returns:
            attended_image_feat: Tensor of shape (B, C, H, W)
        """
        B, C, H, W = image_feat.shape

        Q = self.query_proj(text_feat).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        K = self.key_proj(image_feat)  # (B, H, H, W)
        V = self.value_proj(image_feat)  # (B, H, H, W)

        # Compute attention scores
        attn_scores = (Q * K).sum(dim=1, keepdim=True) / (K.shape[1] ** 0.5)  # (B, 1, H, W)
        attn_weights = F.softmax(attn_scores.view(B, -1), dim=-1).view(B, 1, H, W)  # (B, 1, H, W)

        # Apply attention
        weighted_value = attn_weights * V  # (B, H, H, W)
        attended = self.output_proj(weighted_value)  # (B, C, H, W)
        return attended
    
class EntityGuidedCrossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.feature_dim  # e.g., 640 or 768
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, support_features, entity_vectors, support_labels):
        """
        Args:
            support_features: [N*K, D] image features from support set
            entity_vectors:   [N, D] semantic entity vectors (one per class)
            support_labels:   [N*K] labels for each support image

        Returns:
            refined_support_features: [N*K, D]
        """
        N_K, D, H, W = support_features.shape
        N = entity_vectors.shape[0]  # number of classes

        # Project support features
        support_features = support_features.view(N_K, D, H * W).permute(0, 2, 1)
        support_features = support_features.mean(dim=1)
        K_proj = self.key_proj(support_features)      # [N*K, D]
        V_proj = self.value_proj(support_features)    # [N*K, D]
        
        refined_features = torch.zeros_like(support_features)
        
        for class_idx in range(N):
            class_mask = (support_labels == class_idx)  # [N*K]
            if class_mask.sum() == 0:
                continue

            support_subset = support_features[class_mask]  # [K, D]
            K_subset = K_proj[class_mask]                  # [K, D]
            V_subset = V_proj[class_mask]                  # [K, D]

            # Compute attention with entity vector as Q
            Q = self.query_proj(entity_vectors[class_idx].unsqueeze(0))  # [1, D]
            attn_scores = torch.matmul(Q, K_subset.T) / (D ** 0.5)       # [1, K]
            attn_weights = F.softmax(attn_scores, dim=-1)                # [1, K]
            attended = torch.matmul(attn_weights, V_subset)              # [1, D]

            # Add residual connection
            refined = support_subset + self.out_proj(attended).repeat(support_subset.size(0), 1)  # [K, D]
            refined_features[class_mask] = refined

        return refined_features