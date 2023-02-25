import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SingleLayerGCN(nn.Module):
    def __init__(self, hidden_size):
        super(SingleLayerGCN, self).__init__()
        self.hidden_size = hidden_size
        self.b_iah = nn.Parameter(torch.Tensor(1 * self.hidden_size))
        self.linear_edge_in = nn.Linear(1 * self.hidden_size, 1 * self.hidden_size, bias=False)
        self.linear_edge_out = nn.Linear(1 * self.hidden_size, 1 * self.hidden_size, bias=False)

    def forward(self, adj_out, adj_in, hidden, u):
        h_out = self.linear_edge_out(hidden)
        h_in = self.linear_edge_in(hidden)
        inputs_out = torch.matmul(adj_out, h_out + u) + self.b_iah
        inputs_in = torch.matmul(adj_in.transpose(1, 2), h_in + u)  
        inputs = inputs_in + inputs_out
        return inputs

class SingleDirectGCN(nn.Module):
    def __init__(self, hidden_size, use_gops,ops_emb=None, step=1):
        super(SingleDirectGCN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.use_gops = use_gops
        self.ops_emb = nn.Embedding(102, hidden_size, padding_idx=0) if ops_emb is not None else ops_emb
        self.net = nn.ModuleList([
            SingleLayerGCN(hidden_size) for _ in range(step)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_max_ops_indices(self, seq_inputs, alias_inputs, max_n_node):
        seq_inputs = seq_inputs.unsqueeze(-1)  # b x t x 1
        seq_mask = seq_inputs.bool()  # b x t x 1

        relation_matrix = seq_inputs - seq_inputs.transpose(1, 2)  # b x t x t
        relation_matrix = relation_matrix.bool()  # b x t x t
        relation_matrix = (~relation_matrix).long()
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(1, 2)
        seq_occur_count = relation_matrix.sum(2)  # b x t
        # b x t
        # res = torch.arange(max_n_node, device=seq_inputs.device).unsqueeze(0).expand(seq_inputs.size(0), -1)
        res = torch.zeros([seq_inputs.size(0), max_n_node], dtype=torch.long, device=seq_inputs.device)
        res = res.scatter(dim=-1, index=alias_inputs, src=seq_occur_count)
        return res

    def edge_att(self, A, hidden, ops_emb):
        # w = (hidden * ops_emb).unsqueeze(-2).expand(-1, -1, hidden.size(-2), -1)  # b x t x t x h
        # w = torch.cat([w, A.unsqueeze(-1)], -1)
        # w = self.leaky_relu(self.linear_trans(w))  # b x t x t x h
        # A = w.matmul(self.q)
        B = hidden.bmm(ops_emb.transpose(1, 2))
        return A * B

    def forward(self, adj_out, adj_in, hidden, seq_inputs, alias_inputs, items):
        '''
        A: b x t x t
        hidden: b x t x h
        '''
        if self.use_gops:
            indices = self.get_max_ops_indices(seq_inputs, alias_inputs, items.size(-1))
            w = self.ops_emb(indices)  # b x t x h
            w = F.normalize(w, dim=-1)
        else:
            w = torch.zeros_like(hidden, device=hidden.device)

        for GNNCell in self.net:
            hidden = GNNCell(adj_out, adj_in, hidden, w)
        return hidden
