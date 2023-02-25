import torch
from torch import nn
from torch.nn import Module
import math
import torch.nn.functional as F

from models.GNN import GNN


class SessionGraph(Module):
    def __init__(self, opt, emb_module):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.dim = self.hidden_size
        # self.n_node = n_node
        self.batch_size = opt.batchSize

        # self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding = emb_module  # todo: should not reset when using pretrain model
        # self.add_module('emb', self.embedding)
        self.gnn = GNN(self.hidden_size, opt.nonhybrid, step=opt.step)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def freeze(self, freez):
        for name, child in self.named_children():
            if name == 'embedding':
                for param in child.parameters():
                    param.requires_grad = not freez
            else:
                pass

    def compute_scores(self, seq_hidden, mask, A, items, seq_g_hidden, embs=None):
        if embs is None:
            return self.gnn.compute_scores(seq_hidden, mask, A, items, seq_g_hidden, self.get_embs())
        else:
            return self.gnn.compute_scores(seq_hidden, mask, A, items, seq_g_hidden, embs)

    def forward(self, inputs, A, alias_inputs, targets, mask):
        hidden = self.embedding(inputs)
        g_hidden = self.gnn(A, hidden)
        return hidden, g_hidden, A, inputs

    def get_embs(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight
        return self.embedding.get_weight()


class attnet(nn.Module):
    def __init__(self, hiddenSize, dropout=0.1):
        super().__init__()
        self.hidden_size = hiddenSize
        self.q = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.k = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, mask=None):
        q, k, v = q, q, q
        residual = q
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        attn = torch.matmul(q / (self.hidden_size ** 0.5), k.transpose(1, 2))

        if mask is not None:
            attn = attn * mask.view(mask.shape[0], -1, 1).float()

        attn = self.dropout(F.softmax(attn, dim=-1))
        q = torch.matmul(attn, v)

        q = self.w1(q)
        q = torch.relu(q)
        q = self.w2(q) + residual

        return q, attn


class attnpack(nn.Module):
    def __init__(self, hidden_size, step=3):
        super().__init__()

        self.net = nn.ModuleList([
            attnet(hidden_size) for _ in range(step)
        ])

    def forward(self, hidden, mask):
        for att in self.net:
            hidden, _ = att(hidden, mask)
        return hidden
