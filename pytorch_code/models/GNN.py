import math

import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNNpack(nn.Module):
    def __init__(self, GNN_class, hiddenSize, step):
        super().__init__()
        self.net = nn.ModuleList([
            GNN_class(hiddenSize, 1) for _ in range(step)
        ])
        self.linear_merge = nn.Linear(step*hiddenSize,hiddenSize,bias=True)

    def forward(self, A, hidden):
        hidden_list = []
        for GNN in self.net:
            hidden = GNN(A, hidden)
            hidden_list.append(hidden)
        # hidden = self.linear_merge(torch.cat(hidden_list,dim=-1))
        # return hidden
        return hidden_list[-1]


class GNN(Module):
    def __init__(self, hidden_size, nonhybrid, step=1):
        super(GNN, self).__init__()
        self.nonhybrid = nonhybrid
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

    def compute_scores(self, seq_hidden, mask, A, items, seq_g_hidden, embedding):
        a = self.get_nextItemEmb(seq_hidden, mask, A, items, seq_g_hidden, embedding)
        b = embedding  # n_nodes x latent_size
        # b = self.embedding.get_weight()  # n_nodes x latent_size
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1) * 13
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def get_nextItemEmb(self, seq_hidden, mask, A, items, g_hidden, embedding):
        ht = g_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        return ht
