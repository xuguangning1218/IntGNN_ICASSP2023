import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleDirectGNN(nn.Module):
    def __init__(self, hidden_size, pos_emb, step=1):
        super(SingleDirectGNN, self).__init__()
        self.pos_emb = pos_emb
        self.step = step
        self.hidden_size = hidden_size

        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.linear_edge_in = nn.Linear(self.hidden_size, 1 * self.hidden_size, bias=False)
        # self.linear_gate_i = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        # self.linear_gate_o = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)

        self.reset_parameters()

    def GNNCell(self, A_in, hidden):
        '''
        A: b x t x t
        '''
        h_n = self.linear_edge_in(hidden)
        # h_n = hidden
        # inputs_out = torch.matmul(A_out, h_n) + self.b_iah
        inputs_in = torch.matmul(A_in.transpose(1, 2), h_n) + self.b_oah
        return inputs_in

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_pos_emb(self, seq_mask):
        length = seq_mask.sum(1, keepdim=True)
        coo = torch.arange(seq_mask.size(1), device=seq_mask.device).unsqueeze(0).expand(seq_mask.size(0), -1)
        coo = coo * seq_mask
        pos_index = (length - coo) * seq_mask
        # pos_index = (coo + 1) * seq_mask
        return self.pos_emb(pos_index)

    def pos_graph(self, seq_inputs):
        seq_inputs = seq_inputs.unsqueeze(-1)  # b x t x 1
        seq_mask = seq_inputs.bool()  # b x t x 1

        relation_matrix = seq_inputs - seq_inputs.transpose(1, 2)  # b x t x t
        relation_matrix = relation_matrix.bool()  # b x t x t
        relation_matrix = (~relation_matrix).float().tril(diagonal=0)
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(1, 2)

        # norm = relation_matrix.sum(-1, keepdim=True)  # b x t x 1
        # norm[torch.where(norm == 0)] = 1.
        # A_out = relation_matrix / norm
   
        norm = relation_matrix.sum(-2, keepdim=True)  # b x t x 1
        norm[torch.where(norm == 0)] = 1.
        A_in = relation_matrix / norm
        return A_in  # b x t x t

    def pos_dis(self, seq_mask):
        seq_mask = seq_mask.unsqueeze(-1).long()
        coo = torch.arange(seq_mask.size(-2), device=seq_mask.device).unsqueeze(0).unsqueeze(-1)  # 1 x t x 1
        res = coo - coo.transpose(1, 2)  # 1 x t x t
        res = res.tril(diagonal=0)
        return res.expand(seq_mask.size(0), -1, -1)

    def forward(self, seq_inputs, seq_ori_hidden=None):
        seq_mask = seq_inputs.bool()
        A_in = self.pos_graph(seq_inputs)

        pos_emb = self.get_pos_emb(seq_mask)
        pos_emb = F.normalize(pos_emb)  # b x t x h
 
        for i in range(self.step):
            pos_emb = self.GNNCell(A_in, pos_emb)

        return pos_emb  # b x t x h
