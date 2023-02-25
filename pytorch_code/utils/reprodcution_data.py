import torch

from utils.geo_data import Data


class ReproductionData(Data):
    def __init__(self, data):
        super(ReproductionData, self).__init__(data)

    def __getitem__(self, index):

        inputs, target = self.inputs[index], self.targets[index]
        max_n_node = self.max_len

        inputs = torch.tensor(inputs, dtype=torch.long)  # .cuda()
        node = torch.unique(inputs)
        padding = torch.zeros(size=[max_n_node - len(node)], dtype=torch.long, device=inputs.device)
        items = torch.cat([node, padding])

        relation_matrix = (node.unsqueeze(-1) - inputs.unsqueeze(-1).t()).abs()
        alias_inputs = torch.argmin(relation_matrix, dim=0)
        adj_out, adj_in = self.construct_graph(alias_inputs, max_n_node, inputs.device)

        freq_index = torch.where(inputs==target)
        freq_target = torch.ones(size=(),dtype=torch.long)
        if freq_index[0].size(0)==0:
            freq_target = freq_target*0
        else:
            seq_freq = self.get_freq_position(inputs.unsqueeze(-1),inputs.unsqueeze(-1).bool())
            freq_target = freq_target*seq_freq[freq_index[0][-1]]

        return [alias_inputs, adj_out, adj_in, items, inputs, torch.tensor(target, device=inputs.device), freq_target]
    
    def get_freq_position(self, seq_inputs, seq_mask):
        '''
        seq_inputs: [t,1]
        '''
        relation_matrix = seq_inputs - seq_inputs.transpose(-2, -1)  # t x t
        relation_matrix = relation_matrix.bool()  # t x t
        relation_matrix = (~relation_matrix).long().tril()
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(-2, -1)
        seq_occur_count = relation_matrix.sum(-1).clamp(min=0,max=301)  # t
        return seq_occur_count # [t]

    def construct_graph(self, alias_input, max_n_node, device):
        alias_input = torch.cat([alias_input, torch.zeros([1], dtype=torch.long, device=device)])
        u_A = torch.zeros((max_n_node, max_n_node), dtype=torch.float, device=device)
        coordinte = torch.arange(alias_input.size(0), device=device)
        alias_input_masked_with_coo = alias_input.bool().long() * coordinte
        first_zero_index = alias_input_masked_with_coo.argmax() + 1

        u_A_xindex = alias_input[:first_zero_index - 1]
        u_A_yindex = alias_input[1:first_zero_index]

        if u_A_xindex.size(0) > 0:
            value = torch.ones_like(u_A_xindex, dtype=torch.float)
            u_A.index_put_([u_A_xindex, u_A_yindex], value, accumulate=False)
            # note: weight is different from srgnn when accumulate=True
        u_A = u_A + torch.eye(max_n_node)

        u_sum_out = torch.sum(u_A, -1)
        u_sum_out[torch.where(u_sum_out == 0)] = 1
        u_A_out = torch.divide(u_A.t(), u_sum_out)

        u_sum_in = torch.sum(u_A, -2, keepdim=True)
        u_sum_in[torch.where(u_sum_in == 0)] = 1
        u_sum_in = torch.divide(u_A, u_sum_in)

        return u_A_out.t(), u_sum_in
