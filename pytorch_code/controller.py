#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import numpy as np
import torch


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def fetch_data(i, data):
    '''

    :param i:
    :param data:
    :return: [tensors]
    '''
    alias_inputs_, inputs_, A_, items_, mask_, targets_ = data.get_slice(i)
    mask_ = trans_to_cuda(torch.Tensor(mask_).long())
    targets_ = trans_to_cuda(torch.Tensor(targets_).long())

    return alias_inputs_, items_, A_, mask_, targets_


def get_scores(model, hidden, alias_inputs, mask, targets, A, items, g_hidden):
    index = alias_inputs.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    seq_hidden = torch.gather(hidden, dim=1, index=index)
    seq_g_hidden = torch.gather(g_hidden, dim=1, index=index)

    return targets, model.compute_scores(seq_hidden, mask, A, items, seq_g_hidden)


def test(model, test_data):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(model.batch_size)

    hit10, mrr10, hit20, mrr20 = [], [], [], []

    for i in slices:
        alias_inputs, items, A, mask, targets = fetch_data(i, test_data)
        hidden, g_hidden, A, items = forward(model, items, A, alias_inputs, targets, mask)
        targets, scores = get_scores(model, hidden, alias_inputs, mask, targets, A, items, g_hidden)

        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets.cpu().detach().numpy()):
            hit20.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target)[0][0] + 1))
        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets.cpu().detach().numpy()):
            hit10.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target)[0][0] + 1))

    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100

    return hit10, mrr10, hit20, mrr20


def forward(model, alias_inputs, gru_inputs, A, items, mask, targets):
    # alias_inputs, inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(alias_inputs.long())
    items = trans_to_cuda(items.long())
    A = trans_to_cuda(A.float())
    mask = trans_to_cuda(mask.long())
    hidden = model(items, A, alias_inputs, targets, mask)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data, loss_function, optimizer, scheduler):
    scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    for batch_id, (alias_inputs, gru_inputs, A, items, mask, targets) in enumerate(train_data):
        optimizer.zero_grad()
        targets, scores = forward(model, alias_inputs, gru_inputs, A, items, mask, targets)
        targets = trans_to_cuda(targets.long())
        loss = loss_function(scores, targets - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss
        if batch_id % 100 == 0:
            print('[%d/%d] Loss: %.4f' % (batch_id, train_data.dataset.length / train_data.batch_size, loss.item()))
    # slices = train_data.generate_batch(model.batch_size)
    # for i, j in zip(slices, np.arange(len(slices))):
    #     optimizer.zero_grad()
    #     targets, scores = forward(model, i, train_data)
    #     targets = trans_to_cuda(torch.Tensor(targets).long())
    #     loss = loss_function(scores, targets - 1)
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += loss
    #     if j % int(len(slices) / 5 + 1) == 0:
    #         print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit10, mrr10, hit20, mrr20 = [], [], [], []
    for batch_id, (alias_inputs, gru_inputs, A, items, mask, targets) in enumerate(test_data):
        targets, scores = forward(model, alias_inputs, gru_inputs, A, items, mask, targets)
        targets = trans_to_cpu(targets).detach().numpy()
        mask = trans_to_cpu(mask).detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, mask):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, mask):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))

    # slices = test_data.generate_batch(model.batch_size)
    # for i in slices:
    #     targets, scores = forward(model, i, test_data)
    #     sub_scores = scores.topk(20)[1]
    #     sub_scores = trans_to_cpu(sub_scores).detach().numpy()
    #     for score, target, mask in zip(sub_scores, targets, test_data.mask):
    #         hit20.append(np.isin(target - 1, score))
    #         if len(np.where(score == target - 1)[0]) == 0:
    #             mrr20.append(0)
    #         else:
    #             mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
    #     sub_scores = scores.topk(10)[1]
    #     sub_scores = trans_to_cpu(sub_scores).detach().numpy()
    #     for score, target, mask in zip(sub_scores, targets, test_data.mask):
    #         hit10.append(np.isin(target - 1, score))
    #         if len(np.where(score == target - 1)[0]) == 0:
    #             mrr10.append(0)
    #         else:
    #             mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100
    return hit10, mrr10, hit20, mrr20


def student_forward(teacher, model, i, data, coarse_embedding):
    alias_inputs, inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    inputs = trans_to_cuda(torch.Tensor(inputs).long())

    hidden = model(items, A, alias_inputs, targets, mask)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    hidden_teacher = teacher.gnn(A, model.embedding(items))  # Crucial
    get_t = lambda i: hidden_teacher[i][alias_inputs[i]]
    seq_hidden_teacher = torch.stack([get_t(i) for i in torch.arange(len(alias_inputs)).long()])
    # coarse_embs = teacher.embedding.get_coarse_weight()
    coarse_embs = coarse_embedding.weight
    c_targets = teacher.compute_scores(seq_hidden_teacher, mask,
                                       coarse_embs)  # ['teacher emb x teacher hidden']
    c_targets = c_targets.argmax(dim=1)
    c_scores = teacher.compute_scores(seq_hidden, mask, coarse_embs)  # ['teacher emb x student hidden']
    return targets, model.compute_scores(seq_hidden, mask), c_targets.detach_(), c_scores


def student_train_test(teacher, model, train_data, test_data, loss_function, optimizer, scheduler, coarse_embedding,
                       epoch):
    scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        optimizer.zero_grad()
        targets, scores, c_targets, c_scores = student_forward(teacher, model, i, train_data, coarse_embedding)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = loss_function(scores, targets - 1, c_scores, c_targets, epoch)
        loss.backward()
        optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit10, mrr10, hit20, mrr20 = [], [], [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100
    return hit10, mrr10, hit20, mrr20
