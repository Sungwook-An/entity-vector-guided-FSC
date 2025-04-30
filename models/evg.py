import torch
import torch.nn as nn
import torch.nn.functional as F


class EVGNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_attn_heads=4, top_k=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, class_embedding, entity_embeddings):
        """
        Args:
            class_embedding (Tensor): Shape (D), output from text encoder (E_t)
            entity_embeddings (Tensor): Shape (N, D), entity features from JSON (20 per class)
            
        Returns:
            Tensor of shape (D): final entity representation (v_e)
        """        
        # Project Q, K, V
        Q = self.query_proj(class_embedding).unsqueeze(0) # (1, H)
        K = self.key_proj(entity_embeddings)              # (N, H)
        V = self.value_proj(entity_embeddings)            # (N, H)
        
        # Compute attention
        attn_logits = torch.matmul(Q, K.T) / (self.hidden_dim ** 0.5) # (1, N)
        attn_scores = F.softmax(attn_logits, dim = -1) # (1, N)
        
        # Top-k selection
        k = min(self.top_k, attn_scores.shape[-1])
        
        topk_scores, topk_indices = torch.topk(attn_scores, k, dim=-1)
        topk_indices = topk_indices.squeeze(0) # (top_k)
        topk_scores = topk_scores.squeeze(0)   # (top_k)
        
        # Gather top-k value vectors and and apply weights
        topk_values = V[topk_indices] # (top_k, H)
        weighted_sum = torch.sum(topk_values * topk_scores.unsqueeze(-1), dim=1)
        weighted_sum = torch.sum(topk_values * topk_scores.unsqueeze(-1), dim=0) # (H)
        
        # Final projection
        final_vector = self.output_proj(weighted_sum)
        
        return final_vector
    
class MultiHeadEVGNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_attn_heads=4, top_k=5):
        super().__init__()
        assert hidden_dim % num_attn_heads == 0, "hidden_dim must be divisible by num_attn_heads"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_attn_heads = num_attn_heads
        self.top_k = top_k
        self.head_dim = hidden_dim // num_attn_heads

        # Projection layers
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)

        # Final output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, class_embedding, entity_embeddings):
        """
        Args:
            class_embedding: (D,) or (1, D)
            entity_embeddings: (N, D)
        Returns:
            final_vector: (output_dim,)
        """
        if class_embedding.dim() == 1:
            class_embedding = class_embedding.unsqueeze(0)  # (1, D)

        # Project Q, K, V
        Q = self.query_proj(class_embedding)  # (1, hidden_dim)
        K = self.key_proj(entity_embeddings)  # (N, hidden_dim)
        V = self.value_proj(entity_embeddings)  # (N, hidden_dim)

        # Reshape for multi-head
        Q = Q.view(self.num_attn_heads, self.head_dim)             # (num_attn_heads, head_dim)
        K = K.view(K.size(0), self.num_attn_heads, self.head_dim)   # (N, num_attn_heads, head_dim)
        V = V.view(V.size(0), self.num_attn_heads, self.head_dim)   # (N, num_attn_heads, head_dim)

        # Compute attention for each head
        attn_logits = torch.einsum('hd,nhd->hn', Q, K) / (self.head_dim ** 0.5)  # (num_attn_heads, N)
        attn_scores = F.softmax(attn_logits, dim=-1)  # (num_attn_heads, N)

        # Top-k selection per head
        k = min(self.top_k, attn_scores.shape[-1])
        topk_scores, topk_indices = torch.topk(attn_scores, k, dim=-1)  # (num_attn_heads, top_k)

        # Gather top-k values
        gathered_values = []
        for head in range(self.num_attn_heads):
            selected_V = V[topk_indices[head], head]  # (top_k, head_dim)
            weighted_V = selected_V * topk_scores[head].unsqueeze(-1)  # (top_k, head_dim)
            summed = weighted_V.sum(dim=0)  # (head_dim,)
            gathered_values.append(summed)

        # Concatenate heads
        concat = torch.cat(gathered_values, dim=-1)  # (hidden_dim,)

        # Final output projection
        final_vector = self.output_proj(concat)  # (output_dim,)

        return final_vector