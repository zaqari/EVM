from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

class wv(nn.Module):

    def __init__(self, model: str='roberta-base', device: str='cpu'):
        super(wv, self).__init__()
        self.mod = SentenceTransformer(model, device=device)

    def forward(self, x):
        return self.mod.encode(x, convert_to_tensor=True)