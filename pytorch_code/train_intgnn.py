#!/usr/bin/env python36
# -*- coding: utf-8 -*-


import argparse
import datetime
import gc
import os
import pickle
import random
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from controller import trans_to_cuda, trans_to_cpu
from models.intgnn import ReproductionSessionGraph
from utils.reprodcution_data import ReproductionData
from utils.utils import load_model, n_node
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="Tmall",
    help=
    "dataset name: diginetica/retailRocket_DSAN/Tmall",
)
parser.add_argument("--batchSize", type=int, default=512, help="input batch size")
parser.add_argument("--hiddenSize", type=int, default=100, help="hidden state size")
parser.add_argument(
    "--epoch",
    type=int,
    default=5,
    help="the number of epochs to train for, [digi=12,Tmall=5,Retail=10] ",
)
parser.add_argument(
    "--lr", type=float, default=0.00128, help="learning rate"
)  # [0.001, 0.0005, 0.0001]
parser.add_argument("--lr_dc", type=float, default=0.3, help="learning rate decay rate")
parser.add_argument(
    "--lr_dc_step",
    type=int,
    default=3,
    help="the number of steps after which the learning rate decay",
)
parser.add_argument(
    "--l2", type=float, default=1e-6, help="l2 penalty"
)  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="the number of epoch to wait before early stop ",
)
parser.add_argument(
    "--nonhybrid", action="store_true", help="only use the global preference to predict"
)
parser.add_argument("--gru_dropout", type=float, default=0.0, help="gru dropout")
parser.add_argument("--gnn_dropout", type=float, default=0.0, help="gnn dropout")
parser.add_argument("--emb_dropout", type=float, default=0.2, help="emb dropout")
parser.add_argument("--gru_layer", type=int, default=1, help="gru layer")
parser.add_argument(
    "--delta", type=float, default=12.5, help="norm factor"
)  # 12.0 for separate score loss
parser.add_argument("--use_gpos", action="store_false", help="whether use IP graph")
parser.add_argument("--use_gops", action="store_false", help="whether use IO graph")
parser.add_argument(
    "--use_ops_att", action="store_false", help="whether use recurrence aware attention"
)
parser.add_argument(
    "--use_coarse2fine",
    action="store_false",
    help="whether use occurrence prediction layer",
)
parser.add_argument("--num_worker", type=int, default=4, help="num worker")

opt = parser.parse_args()

random.seed(42)
np.random.seed(42)


def train(model, train_loader, loss_function, optimizer, scheduler):
    logging.info("start training: %s" % datetime.datetime.now())
    print("start training: %s" % datetime.datetime.now())
    model.train()

    total_loss = 0.0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        alias_inputs, adj_out, adj_in, items, inputs, targets, freq_target = [
            x.cuda() for x in data
        ]
        gru_ocur_hidden, sess_emb, scores, max_occur_count, gru_ocur_scores = model.forward(
            alias_inputs, adj_out, adj_in, items, inputs)
        loss = loss_function(scores, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        total_loss += loss.detach().cpu()
    logging.info("\tLoss:\t%.3f" % total_loss)
    print("\tLoss:\t%.3f" % total_loss)
    scheduler.step()

    gc.collect()


def test(model, test_loader):
    logging.info("start predicting: %s" % datetime.datetime.now())
    print("start predicting: %s" % datetime.datetime.now())
    model.eval()

    hit10, mrr10, hit20, mrr20 = [], [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            alias_inputs, adj_out, adj_in, items, inputs, targets, _ = [
                x.cuda() for x in data
            ]
            targets = targets.cpu().numpy()
            gru_ocur_hidden, sess_emb, scores, max_occur_count, gru_ocur_scores = model.forward(
                alias_inputs, adj_out, adj_in, items, inputs)

            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))
            sub_scores = scores.topk(10)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
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


def main(finetune=False):
    
    if not os.path.exists("../result_{}".format(opt.dataset)):
        os.mkdir("../result_{}".format(opt.dataset))
    
    logging.basicConfig(
        filename="../result_{}/intgcn-{}-{}".format(opt.dataset, opt.dataset, finetune) + ".log",
        filemode="w",
        level=logging.INFO,
    )
   
    print(opt)
    
    logging.info("loading data")
    print("loading data")

    train_data_a = pickle.load(open("../datasets/" + opt.dataset + "/train.txt", "rb"))
    test_data_digi = pickle.load(open("../datasets/" + opt.dataset + "/test.txt", "rb"))

    train_data_fintune = ReproductionData(train_data_a)
    train_data_fintune = torch.utils.data.DataLoader(
        train_data_fintune,
        num_workers=opt.num_worker,
        batch_size=opt.batchSize,
        shuffle=True,
        pin_memory=True,
    )
    test_data = ReproductionData(test_data_digi)
    test_data = torch.utils.data.DataLoader(
        test_data,
        num_workers=opt.num_worker,
        batch_size=opt.batchSize,
        shuffle=False,
        pin_memory=True,
    )
    
    logging.info("preparing model")
    print("preparing model")

    model = ReproductionSessionGraph(
        opt,
        nn.Embedding(n_node[opt.dataset] + 1, opt.hiddenSize, padding_idx=0),
    )

    # model = load_model('../result/intgcn-diginetica.pth', model, True)
    model = trans_to_cuda(model)

    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": loss_function.parameters()}],
        lr=opt.lr,
        weight_decay=opt.l2,
    )  #
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc
    )
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 700, 20000)

    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        logging.info("-------------------------------------------------------")
        logging.info("epoch: %d" % epoch)
        print("-------------------------------------------------------")
        print("epoch: %d" % epoch)

        train(model, train_data_fintune, loss_function, optimizer, scheduler)

        hit10, mrr10, hit20, mrr20 = test(model, test_data)
        flag = 0
        if hit20 >= best_result[0]:
            best_result[0] = hit20
            best_epoch[0] = epoch
            flag = 1
        if mrr20 >= best_result[1]:
            best_result[1] = mrr20
            best_epoch[1] = epoch
            flag = 1
        if hit10 >= best_result[2]:
            best_result[2] = hit10
            best_epoch[2] = epoch
            flag = 1
        if mrr10 >= best_result[3]:
            best_result[3] = mrr10
            best_epoch[3] = epoch
            flag = 1

        logging.info("Best Result:")
        logging.info(
            "\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d"
            % (best_result[0], best_result[1], best_epoch[0], best_epoch[1])
        )
        logging.info(
            "\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d,\t%d"
            % (best_result[2], best_result[3], best_epoch[2], best_epoch[3])
        )
        logging.info("Current Result:")
        logging.info("\tRecall@20:\t%.4f\tMRR@20:\t%.4f" % (hit20, mrr20))
        logging.info("\tRecall@10:\t%.4f\tMRR@10:\t%.4f" % (hit10, mrr10))
        print("Best Result:")
        print(
            "\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d"
            % (best_result[0], best_result[1], best_epoch[0], best_epoch[1])
        )
        print(
            "\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d,\t%d"
            % (best_result[2], best_result[3], best_epoch[2], best_epoch[3])
        )
        print("Current Result:")
        print("\tRecall@20:\t%.4f\tMRR@20:\t%.4f" % (hit20, mrr20))
        print("\tRecall@10:\t%.4f\tMRR@10:\t%.4f" % (hit10, mrr10))

        if flag == 1:
            appendix = ""
            if not opt.use_gops:
                appendix = "-woIO"
            torch.save(
                model.state_dict(),
                "../result_{}/intgcn{}-{}-{}.pth".format(opt.dataset, appendix, opt.dataset, finetune),
            )
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    logging.info("-------------------------------------------------------")
    end = time.time()
    logging.info("Run time: %f s" % (end - start))
    print("-------------------------------------------------------")
    print("Run time: %f s" % (end - start))

if __name__ == "__main__":
    main()