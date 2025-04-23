import torch
import torch.nn as nn

from models.resnet12 import Res12
from models.text_encoder import TextEncoder
from models.evg import EVGNetwork
from models.attention_modules import EntityGuidedCrossAttention
from models.classifier import MatchingNetworkClassifier

from modules.prompt_utils import generate_entity_tokens

class INFUSEModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # modules
        self.image_encoder = Res12(**args.image_encoder_config)
        self.text_encoder = TextEncoder(**args.text_encoder_config)
        self.evg = EVGNetwork(**args.evg_config)
        self.cross_attn = EntityGuidedCrossAttention(args)
        self.classifier = MatchingNetworkClassifier(args)
        self.projector = nn.Linear(
            args.text_encoder_config["projection_dim"],
            args.evg_config["output_dim"]
        )
        
        self.args = args
        
        if args.image_encoder_ckpt is not None:
            ckpt = torch.load(args.image_encoder_ckpt)
            
            state_dict = ckpt.get("params", ckpt)
            
            state_dict = self.remove_prefix_from_state_dict(state_dict, "encoder.")
            
            self.image_encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained image encoder from {args.image_encoder_ckpt}")
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, support_images, support_labels, query_images, class_names):
        """
        Args:
            support_images: (N*K, C, H, W)
            support_labels: (N*K)
            query_images:   (N*Q, C, H, W)
            class_names:    List[str] of length N

        Returns:
            logits: (N*Q, N)
        """
        # 1. image encoding
        Z_s = self.image_encoder(support_images)  # [N*K, D]
        Z_q = self.image_encoder(query_images)      # [N*Q, D]

        # 2. text encoding: class_name -> entity tokens -> token embeddings
        entity_vectors = []
        class_embeddings = []
        dataset_name = self.args.dataset
        
        for cls_name in class_names:
            entity_tokens = generate_entity_tokens(dataset_name, cls_name)  # ex: ["fur", "wild", ...]
            token_embeddings = self.text_encoder(entity_tokens)  # [L, D]
            class_embedding = self.text_encoder([cls_name])[0]   # [D]
            entity_vector = self.evg(class_embedding, token_embeddings)  # [D]
            entity_vectors.append(entity_vector)
            class_embedding = self.projector(class_embedding)  # [D]
            class_embeddings.append(class_embedding)

        entity_vectors = torch.stack(entity_vectors, dim=0)  # [N, D]
        # print(entity_vectors.shape) # (5, 640)
        class_embeddings = torch.stack(class_embeddings, dim=0)  # [N, D]
        

        # 3. Cross-attention with entity vector
        ca_result = self.cross_attn(Z_s, entity_vectors, support_labels)
        
        N_K, D, H, W = Z_s.shape
        Z_s = Z_s.view(N_K, D, H * W).permute(0, 2, 1)
        Z_s = Z_s.mean(dim=1)  # [N*K, D]
        F_s_hat = (ca_result * Z_s) + Z_s

        # 4. Matching network classifier
        logits = self.classifier(Z_q, F_s_hat, support_labels)

        return logits
    
    def remove_prefix_from_state_dict(self, state_dict, prefix):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]  # prefix 제거
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict
