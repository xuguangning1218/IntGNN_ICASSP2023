from torch.utils.data import Dataset
import numpy as np
import torch


def data_masks(all_usr_pois, item_tail=None, max_len=100):
    if item_tail is None:
        item_tail = [0]
    new_all_usr_pois = []
    for upois in all_usr_pois:
        if len(upois) > max_len:
            upois = upois[-max_len:]
        new_all_usr_pois.append(upois)
    us_lens = [len(upois) for upois in new_all_usr_pois]
    len_max = max(us_lens) + 1  # note: the last signal must be zero for adapting to graph construction
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(new_all_usr_pois, us_lens)]
    return us_pois, len_max


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity


class Data(Dataset):
    def __init__(self, data):
        inputs, max_len = data_masks(data[0])
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])
        self.max_len = max_len

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

        return [alias_inputs, adj_out, adj_in, items, inputs, torch.tensor(target, device=inputs.device)]

    def __len__(self):
        return self.length

    def ops(self, seq_inputs):
        seq_inputs = seq_inputs.unsqueeze(-1)  # t x 1
        seq_mask = seq_inputs.bool()  # t x 1

        relation_matrix = seq_inputs - seq_inputs.transpose(0, 1)  # t x t
        relation_matrix = relation_matrix.bool()  # t x t
        relation_matrix = (~relation_matrix).long().tril()
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(0, 1)
        seq_occur_count = relation_matrix.sum(-1)  # t

        return seq_occur_count

    def construct_graph(self, alias_input, max_n_node, device):
        alias_input = torch.cat([alias_input, torch.zeros([1], dtype=torch.long, device=device)])
        u_A = torch.zeros((max_n_node, max_n_node), dtype=torch.float, device=device)
        coordinte = torch.arange(alias_input.size(0), device=device)
        alias_input_masked_with_coo = alias_input.bool().long() * coordinte
        first_zero_index = alias_input_masked_with_coo.argmax() + 1

        u_A_xindex = alias_input[:first_zero_index - 1]
        u_A_yindex = alias_input[1:first_zero_index]

        graph_mask = self.ops(u_A_yindex)
        graph_mask = ~(graph_mask - 1).bool()
        if first_zero_index > 1 and (u_A_xindex - u_A_yindex)[0] == 0:
            graph_mask[0] = False

        u_A_xindex = u_A_xindex * graph_mask
        u_A_yindex = u_A_yindex * graph_mask

        if u_A_xindex.size(0) > 0:
            value = torch.ones_like(u_A_xindex, dtype=torch.float)
            u_A.index_put_([u_A_xindex, u_A_yindex], value, accumulate=False)
            # note: weight is different from srgnn when accumulate=True

        u_sum_out = torch.sum(u_A, -1)
        u_sum_out[torch.where(u_sum_out == 0)] = 1
        u_A_out = torch.divide(u_A.t(), u_sum_out)

        return u_A_out.t()
