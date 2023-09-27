import os
import time
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

from model import BPR
from data_utils import TripletUniformPair, load_all
import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed (For reproducability)")
    # Model
    parser.add_argument('--dim',
                        type=int,
                        default=512,
                        help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.025,
                        help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs',
                        type=int,
                        default=10,
                        help="Number of epoch during training")
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="Batch size in one iteration")
    parser.add_argument('--print_every',
                        type=int,
                        default=20,
                        help="Period for printing smoothing loss during training")
    parser.add_argument('--eval_every',
                        type=int,
                        default=1000,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--save_every',
                        type=int,
                        default=10000,
                        help="Period for saving model during training")
    parser.add_argument('--model',
                        type=str,
                        default=os.path.join('/home/phuong/Documents/expohedron/BPR/output', 'bpr.pt'),
                        help="File path for model")
    args = parser.parse_args()
    print(args)
    cudnn.benchmark = True


    ############################## PREPARE DATASET ##########################
    data_inf = load_all()
    user_size, item_size = data_inf['user_size'], data_inf['item_size']
    train_user_list, test_user_list = data_inf['train_user_list'], data_inf["test_user_list"]
    train_pair = data_inf['train_pair']

    ########################### CREATE MODEL #################################
    dataset = TripletUniformPair(item_size, train_user_list, train_pair, True, args.n_epochs)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    model = BPR(user_size, item_size, args.dim, args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    smooth_loss = 0
    idx = 0
    for u, i, j in loader:
        optimizer.zero_grad()
        loss = model(u, i, j)
        loss.backward()
        optimizer.step()
        # writer.add_scalar('train/loss', loss, idx)
        smooth_loss = smooth_loss*0.99 + loss*0.01
        if idx % args.print_every == (args.print_every - 1):
            print('loss: %.4f' % smooth_loss)
        if idx % args.eval_every == (args.eval_every - 1):
            plist, rlist = evaluate.precision_and_recall_k(model.W,
                                                    model.H,
                                                    train_user_list,
                                                    test_user_list,
                                                    klist=[1, 5, 10])
            print('P@1: %.4f, P@5: %.4f P@10: %.4f, R@1: %.4f, R@5: %.4f, R@10: %.4f' % (plist[0], plist[1], plist[2], rlist[0], rlist[1], rlist[2]))
            # writer.add_scalars('eval', {'P@1': plist[0],
            #                                         'P@5': plist[1],
            #                                         'P@10': plist[2]}, idx)
            # writer.add_scalars('eval', {'R@1': rlist[0],
            #                                     'R@5': rlist[1],
            #                                     'R@10': rlist[2]}, idx)
        if idx % args.save_every == (args.save_every - 1):
            dirname = os.path.dirname(os.path.abspath(args.model))
            os.makedirs(dirname, exist_ok=True)
            torch.save(model.state_dict(), args.model)
        idx += 1

    dirname = os.path.dirname(os.path.abspath(args.model))
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), args.model)