from tkinter import NONE
import torch
import torch.nn.functional as F
from torch import nn

from models.sessionGraph import SessionGraph
from models.singleDirectGCN import SingleDirectGCN
from models.singleDirectGNN import SingleDirectGNN
from utils.utils import n_node


class ReproductionSessionGraph(SessionGraph):

    def __init__(self, opt,
                 emb_module):  #, dataset='diginetica',pretrained_items=None
        super(ReproductionSessionGraph, self).__init__(opt, emb_module)
        del self.gnn
        self.w = opt.delta
        self.emb_dropout = opt.emb_dropout
        self.gnn_dropout = opt.gnn_dropout
        self.use_gpos = opt.use_gpos
        self.use_gops = opt.use_gops
        self.use_ops_att = opt.use_ops_att
        self.use_coarse2fine = opt.use_coarse2fine
        # self.n_node = n_node[dataset]
        # ----------GNN-------------------------
        self.ops_emb = nn.Embedding(301, opt.hiddenSize)
        self.pos_emb = nn.Embedding(301, opt.hiddenSize, padding_idx=0)
        self.explore_gnn = SingleDirectGCN(self.hidden_size, self.ops_emb,
                                           opt.use_gops, opt.step)
        self.pos_gnn = SingleDirectGNN(self.hidden_size, self.pos_emb,
                                       1)  # opt.step
        # ----------ATT and simple merge---------
        self.linear_trans = nn.Linear(self.hidden_size,
                                      self.hidden_size,
                                      bias=True)
        self.linear_merge = nn.Linear(self.hidden_size,
                                      self.hidden_size,
                                      bias=True)
        self.q = nn.Parameter(torch.rand(size=[1 * self.hidden_size, 1]))
        # ----------score attention---------------
        self.gru = nn.GRU(self.hidden_size,
                          self.hidden_size,
                          num_layers=1,
                          dropout=0.1,
                          batch_first=True,
                          bidirectional=True)
        self.linear_gru = nn.Linear(2 * self.hidden_size,
                                    self.hidden_size,
                                    bias=False)
        self.reset_parameters()
        # ----------donnt reset pretrain params---------------
        # self.embedding_cates = None
        # if pretrained_items is not None:
        #     self.embedding_cates = yjyEmbbeding2(n_node[opt.dataset] , opt.hiddenSize,None,_weight=pretrained_items)
        #     self.embedding_cates.weight.requires_grad = False

    def get_ops_emb(self, seq_inputs):
        seq_inputs = seq_inputs.unsqueeze(-1)  # b x t x 1
        seq_mask = seq_inputs.bool()  # b x t x 1

        relation_matrix = seq_inputs - seq_inputs.transpose(1, 2)  # b x t x t
        relation_matrix = relation_matrix.bool()  # b x t x t
        relation_matrix = (~relation_matrix).long().tril()
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(1, 2)
        seq_occur_count = relation_matrix.sum(2)  # b x t
        return self.ops_emb(seq_occur_count), seq_occur_count  # b x t x h

    def get_max_occ(self, seq_inputs):
        seq_inputs = seq_inputs.unsqueeze(-1)  # b x t x 1
        seq_mask = seq_inputs.bool()  # b x t x 1

        relation_matrix = seq_inputs - seq_inputs.transpose(1, 2)  # b x t x t
        relation_matrix = relation_matrix.bool()  # b x t x t
        relation_matrix = (~relation_matrix).long()
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(1, 2)
        max_occur_count = relation_matrix.sum(2)  # b x t
        return max_occur_count  # b x t

    def contrasive_logit(self, emb_a, emb_b):
        emb_c = torch.cat([emb_a[1:], emb_a[0].unsqueeze(0)], dim=0)
        a = F.normalize(emb_a, dim=-1)
        b = F.normalize(emb_b, dim=-1)
        c = F.normalize(emb_c, dim=-1)

        postive_score = a.mul(b).sum(-1)
        negative_score = a.mul(c).sum(-1)

        logit = torch.stack([postive_score, negative_score], dim=1)  # [b,2]
        return logit * 4

    def forward(self,
                alias_inputs,
                adj_out,
                adj_in,
                items,
                seq_inputs,
                target_freq=None):
        seq_mask = seq_inputs.bool().unsqueeze(-1)
        seq_len = seq_mask.sum(-2)
        seq_ori_hidden = self.embedding(seq_inputs)
        seq_ori_hidden = F.dropout(seq_ori_hidden,
                                   self.emb_dropout,
                                   training=self.training)
        # if self.embedding_cates is None:
        #     seq_cate_hidden = torch.zeros_like(seq_ori_hidden,requires_grad=False)
        # else:
        #     seq_cate_hidden = self.embedding_cates(seq_inputs)
        # -----------gnn--------------
        if self.use_gpos:
            pos_emb = self.pos_gnn(seq_inputs, seq_ori_hidden)
        # print(items)
        u_items_hidden = self.embedding(items)
        u_items_hidden = F.dropout(u_items_hidden,
                                   self.emb_dropout,
                                   training=self.training)
        # if self.embedding_cates is None:
        #     u_cate_hidden = torch.zeros_like(u_items_hidden,requires_grad=False)
        # else:
        #     u_cate_hidden = self.embedding_cates(items)
        explore_hidden = self.explore_gnn(adj_out, adj_in, u_items_hidden,
                                          seq_inputs, alias_inputs, items)
        explore_hidden = explore_hidden.gather(
            dim=-2,
            index=alias_inputs.unsqueeze(-1).expand(-1, -1,
                                                    explore_hidden.size(-1)))
        exp_before = explore_hidden
        exp_after = exp_before
        # -----------att--------------
        seq_occur_hidden, seq_occur_count = self.get_ops_emb(seq_inputs)
        if self.use_ops_att:
            w = F.normalize(seq_occur_hidden, dim=-1)
            explore_hidden = torch.einsum(
                'bth,bth,btg->btg',
                self.linear_trans(explore_hidden).sigmoid(), w, explore_hidden)
            exp_after = explore_hidden
        # -----------merge--------------
        seq_merge = (pos_emb +
                     explore_hidden) if self.use_gpos else explore_hidden
        seq_merge = self.linear_merge(seq_merge)
        q = F.normalize(self.q, dim=0)
        alpha = seq_merge.sigmoid().matmul(q) * seq_mask
        # -----------readout-----------
        seq_ori_hidden = F.normalize(seq_ori_hidden, p=2, dim=-1)
        sess_emb = (alpha * seq_mask).transpose(
            1, 2).bmm(seq_ori_hidden).squeeze(1)
        # -----------compute score-----------
        score = self.compute_scores(sess_emb, self.embedding.weight)
        score = score * self.w
        if not self.use_coarse2fine:
            return None, score

        gru_ocur_hidden, _ = self.gru(seq_occur_hidden)
        gru_ocur_hidden = gru_ocur_hidden * seq_mask
        gru_ocur_hidden = self.linear_gru(gru_ocur_hidden.sum(1) / seq_len)
        gru_ocur_scores = self.compute_scores(gru_ocur_hidden,
                                              self.ops_emb.weight)  # [b,301]
        score_att_mask = torch.zeros([seq_inputs.size(0), 301],
                                     dtype=torch.long,
                                     device=seq_inputs.device)

        score_att_mask = score_att_mask.scatter(dim=-1,
                                                index=seq_occur_count,
                                                value=1).bool()  # [b,301]
        gru_ocur_scores = gru_ocur_scores.masked_fill(~score_att_mask,
                                                      float('-1'))

        max_occur_count = self.get_max_occ(seq_inputs)
        zeros = torch.zeros_like(score, dtype=torch.long)
        scatterd_occ = zeros.scatter(dim=-1,
                                     index=seq_inputs,
                                     src=seq_occur_count)
        scatterd_score = gru_ocur_scores.gather(dim=-1, index=scatterd_occ)

        score = scatterd_score + score
        return gru_ocur_hidden, sess_emb, score, max_occur_count, gru_ocur_scores  # (exp_before,exp_after,sess_emb)

    def get_attention_value(self, seq_inputs, explore_hidden=None):
        if explore_hidden is None:
            return None  # todo
        seq_mask = seq_inputs.bool().unsqueeze(-1)
        seq_ori_hidden = self.embedding(seq_inputs)
        if self.use_gpos:
            pos_emb = self.pos_gnn(seq_inputs, seq_ori_hidden)
        seq_merge = (pos_emb +
                     explore_hidden) if self.use_gpos else explore_hidden
        seq_merge = self.linear_merge(seq_merge)
        q = F.normalize(self.q, dim=0)
        alpha = seq_merge.sigmoid().matmul(q) * seq_mask
        return alpha

    def compute_scores(self, x, A):
        # A [items x hidden]
        x = F.normalize(x, p=2, dim=1)
        A = F.normalize(A, p=2, dim=1)
        scores = torch.matmul(x, A.t())  # batch x items
        return scores
