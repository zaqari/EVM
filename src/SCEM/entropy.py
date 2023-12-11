import torch
import numpy as np
import torch.nn as nn

class entropy(nn.Module):
    
    def __init__(self, sigma=.8, dim=None):
        super(entropy, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.N = torch.distributions.Normal(1,scale=sigma, validate_args=False)
        self.dim = dim

    def forward(self, ex, ey):

        # Get cosine similarity comparison between lexical items
        C = self.cos(ex.unsqueeze(1), ey)

        if self.dim != None:
            #along a single dimension,
            # (1) Get max cosine similarity
            # (2) Get log prob of similarity
            # (3) Calculate log prob and entropy
            C = self.N.log_prob(1-C.max(dim=self.dim).values)
            return -(torch.exp(C) * C).sum()

        else:
            C1, C2 = self.N.log_prob(1-C.max(dim=-1).values), self.N.log_prob(1-C.max(dim=0).values)
            C = None
            return -(torch.exp(C1) * C1).sum(), -(torch.exp(C2) * C2).sum()

    def unsummed(self, ex, ey, dim=None):
        if bool(dim):
            self.dim=dim
        else:
            self.dim=-1

        C = self.cos(ex.unsqueeze(1), ey)

        C = self.N.log_prob(C.max(dim=self.dim).values)
        return -(torch.exp(C) * C)