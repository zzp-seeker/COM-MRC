import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ConditionalLayerNorm(nn.Module):
    def __init__(self, features,condition,eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()

        self.beta_ffn = nn.Linear(condition,features,bias=False)
        torch.nn.init.constant_(self.beta_ffn.weight, 0.)
        self.beta = nn.Parameter(torch.zeros(features))

        self.gamma_ffn = nn.Linear(condition,features,bias=False)
        torch.nn.init.constant_(self.gamma_ffn.weight, 0.)
        self.gamma = nn.Parameter(torch.ones(features))

        self.eps = eps

    def forward(self, x, c):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        beta = self.beta + self.beta_ffn(c)
        gamma = self.gamma + self.gamma_ffn(c)

        return gamma * (x - mean) / (std + self.eps) + beta

