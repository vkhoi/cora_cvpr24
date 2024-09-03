# Implementation based on: https://arxiv.org/pdf/2105.14491

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GraphAttention(nn.Module):
    def __init__(
            self, 
            n_layers=2, 
            d=512, 
            n_heads=16,
            dropout=0.0,
            use_residual=True,
        ):
        """Initializes graph attention network.

        Args:
        * n_layers: number of layers.
        * d: hidden dimension.
        * n_heads: number of heads (since graph attention network also implements multi-head attention).
        * dropout: dropout rate.
        * use_residual: whether to use residual connection for output embeddings.
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_residual = use_residual

        self.fc = nn.ModuleList()
        self.fc2 = nn.ModuleList()
        self.fc_val = nn.ModuleList()
        self.fc_aff = nn.ParameterList()
        self.fc_aff_bias = nn.ParameterList()
        self.lnorms = nn.ModuleList()

        for i in range(n_layers):
            self.lnorms.append(nn.LayerNorm(d))
            self.fc.append(nn.Linear(d, d))
            self.fc2.append(nn.Linear(d, d))
            self.fc_val.append(nn.Sequential(
                nn.Linear(d, d),
                nn.Dropout(dropout),
            ))

            # Make each affinity FC layer different among different heads
            aff = nn.Parameter(
                torch.empty((n_heads, d // n_heads), dtype=torch.float)
            )
            aff_bias = nn.Parameter(
                torch.empty((n_heads,), dtype=torch.float)
            )
            self.fc_aff.append(aff)
            self.fc_aff_bias.append(aff_bias)

        self.init_weights()

    def init_weights(self):
        for i in range(self.n_layers):
            for j in range(self.n_heads):
                init.kaiming_uniform_(self.fc_aff[i][j:j+1], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc_aff[i][:1])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.fc_aff_bias[i], -bound, bound)

    def forward(self, x, adj, t=0):
        # x: (*, n, d) node embeddings
        # adj: (*, n, n) adjacency matrix

        x_orig_shape = x.shape

        bs, n, d = x.shape
        device = x.device
        dd = d // self.n_heads

        x_val = self.fc_val[t](x)
        x_val = x_val.view(bs, n, self.n_heads, dd)
        x_val = x_val.permute(0, 2, 1, 3) # (bs, n_heads, n, d')

        x_aff = self.fc[t](x) # (bs, n, d)
        x_aff = x_aff.view(bs, n, self.n_heads, dd) # (bs, n, n_heads, dd)
        x_aff = x_aff.permute(0, 2, 1, 3) # (bs, n_heads, n, dd)
        
        x_aff2 = self.fc2[t](x) # (bs, n, d)
        x_aff2 = x_aff2.view(bs, n, self.n_heads, dd) # (bs, n, n_heads, dd)
        x_aff2 = x_aff2.permute(0, 2, 1, 3) # (bs, n_heads, n, dd)

        inds = torch.arange(0, n, device=device)
        inds = torch.cartesian_prod(inds, inds) # (nxn, 2)
        inds = inds.unsqueeze(0).expand(bs, self.n_heads, -1, -1) # (bs, n_heads, nxn, 2)

        v_a = torch.gather(x_aff, dim=2, index=inds[:,:,:,0].unsqueeze(-1).expand(-1, -1, -1, dd)) # (bs, n_heads, nxn, dd)
        v_b = torch.gather(x_aff2, dim=2, index=inds[:,:,:,1].unsqueeze(-1).expand(-1, -1, -1, dd)) # (bs, n_heads, nxn, dd)
        v = v_a + v_b # (bs, n_heads, nxn, dd)
        v = v.permute(0, 2, 1, 3) # (bs, nxn, n_heads, dd)
        v = F.leaky_relu(v, negative_slope=0.2)
        affinity = (self.fc_aff[t].view(1, 1, self.n_heads, -1) * v).sum(-1) # (bs, nxn, n_heads)
        affinity = affinity + self.fc_aff_bias[t].view(1, 1, -1)
        affinity = affinity.permute(0, 2, 1)
        affinity = affinity.view(bs, self.n_heads, n, n) # (bs, n_heads, n, n)
        
        masked = (adj < 1e-5)
        masked = masked.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        affinity[masked] = -1e9
        affinity = F.softmax(affinity, dim=-1)

        if self.use_residual:
            x_val = torch.matmul(affinity, x_val) # (bs, n_heads, n, d')
            x_val = x_val.permute(0, 2, 1, 3).flatten(2, 3) # (bs, n, d)
            x_aff = x_aff.permute(0, 2, 1, 3).flatten(2, 3) # (bs, n, d)
            x = self.lnorms[t](x_aff + x_val)
            x = F.relu(x)
        else:
            x = torch.matmul(affinity, x_val) # (bs, n_heads, n, d')
            x = x.permute(0, 2, 1, 3).flatten(2, 3) # (bs, n, d)
            x = self.lnorms[t](x)
            x = F.relu(x)

        x = x.contiguous() # (bs, n, d)
        x = x.view(*x_orig_shape)

        return x
