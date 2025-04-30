import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, BertTokenizer, BertModel


class TextEncoder(nn.Module):
    def __init__(self, encoder_name='clip-vit', projection_dim=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.encoder_name = encoder_name.lower()
        self.device = device

        if self.encoder_name == 'clip-vit':
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
            # self.output_dim = 512

        elif self.encoder_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
            # self.output_dim = 768
        
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        self.output_dim = self.model.config.hidden_size

        self.model.to(self.device)
        self.model.eval()  # Inference mode by default
        
        if projection_dim is not None:
            self.projector = nn.Linear(self.output_dim, projection_dim).to(self.device)
            self.output_dim = projection_dim
        else:
            self.projector = None

    def forward(self, text_list):
        """
        Args:
            text_list (List[str]): A batch of input strings

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            if self.encoder_name == 'clip-vit':
                text_features = outputs.last_hidden_state[:, 0, :]  # CLS token
            elif self.encoder_name == 'bert':
                text_features = outputs.pooler_output  # [CLS] pooled
        
        if self.projector:
            text_features = self.projector(text_features)
            
        return text_features
