# Source based on: https://github.com/woodfrog/vse_infty/blob/master/lib/modules/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    "multiple layer fully-connection, alse called feedforward network"

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h)
        self.bns.append(nn.Identity())

    def forward(self, x):
        B, N, D = x.size()
        x = x.view(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x
