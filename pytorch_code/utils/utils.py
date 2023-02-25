#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import math
import os
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils

from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import seaborn as sns

n_node = {
    "diginetica": 43098 - 1,
    "yoochoose1_64": 37484 - 1,
    "yoochoose1_4": 37484 - 1,
    "Tmall": 40728 - 1,
    "Nowplaying": 60416,
    "retailRocket": 46874 - 1,
    "retailRocket_DSAN": 36969 - 1,
    "sample": 310 - 1,
    "sample_retail": 301 - 1,
    "sample_nowplaying": 5000,
}


def load_model(path, empty_model, freeze_emb=False):
    if not os.path.exists(path):
        return empty_model
    teacher_dict = torch.load(path)
    state_dict = empty_model.state_dict()
    state_dict.update(teacher_dict)
    empty_model.load_state_dict(state_dict)
    if freeze_emb:
        for k, v in empty_model.named_parameters():
            if k.find("embedding.weight") > -1:
                v.requires_grad = False
    return empty_model


def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param batch_data: 元组，第一个元素：句子序列数据，第二个元素：mask，第二个元素：target
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), tensor([1, 1, 0, 0]), 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    # batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    sent_seq = [xi[0] for xi in batch_data]
    padded_sent_seq = rnn_utils.pad_sequence(
        sent_seq, batch_first=True, padding_value=0
    )

    items = []
    A = []
    alias_inputs = []
    for u_input in sent_seq:
        node = torch.unique(u_input)
        items.append(node)
        u_A = np.zeros((padded_sent_seq.size(1), padded_sent_seq.size(1)))
        for i in np.arange(u_input.size(0) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    alias_inputs = np.asarray(alias_inputs)
    A = np.asarray(A)
    delta = padded_sent_seq.size(1) - items[0].size(0)
    if delta > 0:
        items[0] = torch.cat([items[0], torch.zeros(delta)])
    padded_items = rnn_utils.pad_sequence(items, batch_first=True, padding_value=0)
    # return padded_sent_seq.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32)
    return (
        torch.tensor(alias_inputs),
        torch.stack(sent_seq, dim=0),
        torch.tensor(A),
        padded_items,
        torch.stack([xi[1] for xi in batch_data], dim=0),
        torch.stack([xi[2] for xi in batch_data], dim=0).long(),
    )


def data_masks(all_usr_pois, item_tail=None):
    if item_tail is None:
        item_tail = [0]
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = (
        max(us_lens) + 1
    )  # note: the last signal must be zero for adapting to graph construction
    us_pois = [
        upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)
    ]
    gru_inputs = [
        list(reversed(upois)) + item_tail * (len_max - le)
        for upois, le in zip(all_usr_pois, us_lens)
    ]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, gru_inputs, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype="int32")
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1.0 - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data:
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, gru_inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.gru_inputs = np.asarray(gru_inputs)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][: (self.length - batch_size * (n_batch - 1))]
        return slices

    def construct_alias_inputs(self, u_input, node=None):
        if node is None:
            node = torch.unique(u_input)
        relation_matrix = (node.unsqueeze(-1) - u_input.unsqueeze(-1).t()).abs()
        alias_input = torch.argmin(relation_matrix, dim=0)

        return alias_input, node

    def construct_graph(self, alias_input, max_n_node, device):
        alias_input = torch.cat(
            [alias_input, torch.zeros([1], dtype=torch.long, device=device)]
        )
        u_A = torch.zeros((max_n_node, max_n_node), dtype=torch.float, device=device)
        # first_zero_index = torch.argmin(alias_input)
        coordinte = torch.arange(alias_input.size(0), device=device)
        alias_input_masked_with_coo = alias_input.bool().long() * coordinte
        first_zero_index = alias_input_masked_with_coo.argmax() + 1

        u_A_xindex = alias_input[: first_zero_index - 1]
        u_A_yindex = alias_input[1:first_zero_index]
        if u_A_xindex.size(0) > 0:
            value = torch.ones_like(u_A_xindex, dtype=torch.float)
            u_A.index_put_([u_A_xindex, u_A_yindex], value, accumulate=False)
            # note: weight is different from srgnn when accumulate=True

        u_sum_in = torch.sum(u_A, 0)
        u_sum_in[torch.where(u_sum_in == 0)] = 1
        u_A_in = torch.divide(u_A, u_sum_in)
        u_sum_out = torch.sum(u_A, 1)
        u_sum_out[torch.where(u_sum_out == 0)] = 1
        u_A_out = torch.divide(u_A.t(), u_sum_out)
        u_A = torch.cat([u_A_in, u_A_out]).t()
        return u_A

    def _proceed_yjy(self, i, max_n_node, inputs):
        gpu = torch.device("cuda")
        with torch.no_grad():
            # u_input = torch.cuda.LongTensor(inputs[i])
            try:
                u_input = torch.tensor(inputs[i], dtype=torch.long).cuda()
            except Exception as e:
                print(e)
            # if torch.min(u_input) > 0: u_input = torch.cat([u_input, torch.zeros(size=[1]).long().cuda()])
            node = torch.unique(u_input)
            padding = torch.zeros(
                size=[max_n_node - len(node)], dtype=torch.long, device=gpu
            )
            item = torch.cat([node, padding])
            # alias_input = [np.where(node == j)[0][0] for j in u_input]
            relation_matrix = (node.unsqueeze(-1) - u_input.unsqueeze(-1).t()).abs()
            alias_input = torch.argmin(relation_matrix, dim=0)

            u_A = self.construct_graph(alias_input, max_n_node, gpu)

        return i, alias_input, u_A, item

    def _proceed_cpu(self, i, max_n_node, inputs):
        u_input = inputs[i]
        node = np.unique(u_input)
        item = node.tolist() + (max_n_node - len(node)) * [0]

        u_A = np.zeros((max_n_node, max_n_node))
        for j in np.arange(len(u_input) - 1):
            if u_input[j + 1] == 0:
                break
            u = np.where(node == u_input[j])[0][0]
            v = np.where(node == u_input[j + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        alias_input = [np.where(node == j)[0][0] for j in u_input]
        return (
            i,
            torch.cuda.LongTensor(alias_input),
            torch.cuda.FloatTensor(u_A),
            torch.cuda.LongTensor(item),
        )

    def _proceed(self, i, max_n_node, inputs):
        u_input = inputs[i]
        node = np.unique(u_input)
        item = node.tolist() + (max_n_node - len(node)) * [0]

        u_A = np.zeros((max_n_node, max_n_node))
        for j in np.arange(len(u_input) - 1):
            if u_input[j + 1] == 0:
                break
            u = np.where(node == u_input[j])[0][0]
            v = np.where(node == u_input[j + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        alias_input = [np.where(node == j)[0][0] for j in u_input]

        return i, alias_input, u_A, item

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        gru_inputs = self.gru_inputs[i]
        length = len(inputs)
        items, n_node, A, alias_inputs = (
            [None] * length,
            [],
            [None] * length,
            [None] * length,
        )
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        threads = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for j in range(length):
                t = executor.submit(self._proceed_yjy, j, max_n_node, inputs)
                threads.append(t)
            for t in as_completed(threads):
                j, alias_input, u_A, item = t.result()
                alias_inputs[j] = alias_input
                A[j] = u_A
                items[j] = item
        return (
            torch.stack(alias_inputs),
            gru_inputs,
            torch.stack(A),
            torch.stack(items),
            mask,
            targets,
        )


class CompetitiveDataset:
    def __init__(self, data_list, shuffle=False):
        self.data_process_list = []
        length = 0
        max_length_i = 0
        for i in range(len(data_list)):
            data_i = Data(data_list[i], shuffle)
            self.data_process_list.append(data_i)
            if length < data_i.length:
                length = data_i.length
                max_length_i = i
        self.drange = range(len(self.data_process_list))
        self.length = length
        self.shuffle = shuffle
        self.max_length_i = max_length_i
        self.alias_inputs_l = []
        self.gru_inputs_l = []
        self.A_l = []
        self.items_l = []
        self.mask_l = []
        self.targets_l = []

    def generate_batch(self, batch_size):
        for i in range(len(self.data_process_list)):
            if i == self.max_length_i:
                continue
            self.data_process_list[i].generate_batch(batch_size)
        return self.data_process_list[self.max_length_i].generate_batch(batch_size)

    def _proceed(self, x, i):
        if x == self.max_length_i:
            alias_inputs, gru_inputs, A, items, mask, targets = self.data_process_list[
                x
            ].get_slice(i)
            return x, alias_inputs, gru_inputs, A, items, mask, targets
        else:
            length = self.data_process_list[x].length
            fake_i = i
            if np.max(i) >= length:
                fake_i = np.random.randint(low=0, high=length, size=len(i))
            alias_inputs, gru_inputs, A, items, mask, targets = self.data_process_list[
                x
            ].get_slice(fake_i)
            return x, alias_inputs, gru_inputs, A, items, mask, targets

    def get_slice(self, i):
        self.alias_inputs_l = [None] * len(self.data_process_list)
        self.gru_inputs_l = [None] * len(self.data_process_list)
        self.A_l = [None] * len(self.data_process_list)
        self.items_l = [None] * len(self.data_process_list)
        self.mask_l = [None] * len(self.data_process_list)
        self.targets_l = [None] * len(self.data_process_list)
        threads = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            for x in self.drange:
                t = executor.submit(self._proceed, x, i)
                threads.append(t)
            for t in as_completed(threads):
                j, alias_inputs, gru_inputs, A, items, mask, targets = t.result()
                self.alias_inputs_l[j] = alias_inputs
                self.gru_inputs_l[j] = gru_inputs
                self.A_l[j] = A
                self.items_l[j] = items
                self.mask_l[j] = mask
                self.targets_l[j] = targets

        return (
            self.alias_inputs_l,
            self.gru_inputs_l,
            self.A_l,
            self.items_l,
            self.mask_l,
            self.targets_l,
        )

    def _get_specific_slice(self, i, dataset_index):
        return self.data_process_list[dataset_index].get_slice(i)


class MergeData(Data):
    @classmethod
    def trans(self, data2, map_dict=None, offset=43098):
        # inputs2 = np.asarray(data2[0]) + offset
        # np.where((inputs2 - offset) in map_dict, map_dict[inputs2 - offset], inputs2)
        for i in range(len(data2[0])):
            for j in range(len(data2[0][i])):
                data2[0][i][j] = data2[0][i][j] + offset
        # for i in range(len(data2[0])):
        #     for j in range(len(data2[0][i])):
        #         if (data2[0][i][j] - offset) in map_dict:
        #             data2[0][i][j] = map_dict[data2[0][i][j] - offset]
        for i in range(len(data2[1])):
            data2[1][i] = data2[1][i] + offset
        # for i in range(len(data2[1])):
        #     if (data2[1][i] - offset) in map_dict:
        #         data2[1][i] = map_dict[data2[1][i] - offset]
        # todo: it is better to specify merged indexes;
        #  for example, map_dict[5555]=1 , the embedding[5555] should be masked out during training and testing
        return data2[0], data2[1]

    def __init__(
        self, data, data2, map_dict=None, offset=None, shuffle=False, graph=None
    ):
        inputs = data[0]
        targets = data[1]
        offset = len(inputs) if offset is None else offset
        inputs2, targets2 = MergeData.trans(data2, map_dict, offset)
        inputs = inputs + inputs2
        targets = targets + targets2
        super(MergeData, self).__init__([inputs, targets], shuffle, graph)


class ForkData(CompetitiveDataset):
    """
    this class will offset items id accordingly
    """

    def __init__(self, data_list, n_node_list, shuffle=False):
        super(ForkData, self).__init__(data_list, shuffle)
        self.n_node_list = n_node_list

    def get_slice(self, indexes):
        """
        TODO: gru_inputs_l is not correct!
        :param indexes:
        :return:
        """
        (
            alias_inputs_l,
            gru_inputs_l,
            A_l,
            items_l,
            mask_l,
            targets_l,
        ) = super().get_slice(indexes)
        sum_len = 0
        for i in range(len(targets_l)):
            tmp = items_l[i] + sum_len
            zeros = torch.zeros_like(tmp, device=tmp.device)
            items_l[i] = torch.where(items_l[i] == 0, zeros, tmp)

            tmp = targets_l[i] + sum_len
            zeros = np.zeros_like(tmp)
            targets_l[i] = np.where(targets_l[i] == 0, zeros, tmp)

            sum_len = sum_len + self.n_node_list[i]
        return alias_inputs_l, gru_inputs_l, A_l, items_l, mask_l, targets_l


def save_ablo(path, data, data2, data3, data4):
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=[40, 9])
    plt.xticks(fontsize=72)
    plt.yticks(fontsize=72)

    plt.subplot(141)
    ax = sns.barplot(x=data["x"], y=data["y"], hue=data["class"])
    ax.set_ylim((50.5, 55.5))
    # ax.set_ylim((60, 64))
    # ax.set_ylabel('performance (%)', fontsize=24)
    ax.legend(prop=dict(size=18))
    # ax.set_xticklabels('P20', fontsize=14)
    # subplt.yticks(fontsize=72)
    # ax.set_yticks(range(44, 58))
    # ax.set_yticklabels(range(4))
    # ax.set(title="precision of top 20 and top 10 among different model")
    # plt.ylabel('probability')

    plt.subplot(142)
    ax = sns.barplot(x=data2["x"], y=data2["y"], hue=data2["class"])
    ax.set_ylim((37.5, 42.0))
    ax.legend(prop=dict(size=18))
    # ax.set_ylim((53, 56))
    # ax.set_ylabel('performance (%)', fontsize=24)

    plt.subplot(143)
    ax = sns.barplot(x=data3["x"], y=data3["y"], hue=data3["class"])
    ax.set_ylim((17.0, 19.5))
    ax.legend(prop=dict(size=18))
    # ax.set_ylim((34, 37))
    # ax.set_ylabel('performance (%)', fontsize=24)

    plt.subplot(144)
    ax = sns.barplot(x=data4["x"], y=data4["y"], hue=data2["class"])
    ax.set_ylim((16.0, 18.5))
    ax.legend(prop=dict(size=18))
    # ax.set_ylim((34, 37))
    # ax.set_ylabel('performance (%)', fontsize=24)

    savefig(path)
    plt.close()
