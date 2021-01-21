import torch
import math
import numpy as np
import pandas as pd

import os
import json

from models.dynamic_net import Vcnet, Drnet, TR
from data.data import get_iter
from utils.eval import curve

import argparse

def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, model_name='', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu1/eval', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu1/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=100, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    args = parser.parse_args()

    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # data
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    Result = {}
    for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr']:
        Result[model_name]=[]
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
            cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        # use Target Regularization
        if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
            isTargetReg = 1
        else:
            isTargetReg = 0

        if isTargetReg:
            tr_knots = list(np.arange(0.1, 1, 0.1))
            tr_degree = 2
            TargetReg = TR(tr_degree, tr_knots)
            TargetReg._initialize_weights()

        # best cfg for each model
        if model_name == 'Tarnet':
            init_lr = 0.05
            alpha = 1.0

            Result['Tarnet'] = []

        elif model_name == 'Tarnet_tr':
            init_lr = 0.05
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.

            Result['Tarnet_tr'] = []

        elif model_name == 'Drnet':
            init_lr = 0.05
            alpha = 1.

            Result['Drnet'] = []

        elif model_name == 'Drnet_tr':
            init_lr = 0.05
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.

            Result['Drnet_tr'] = []

        elif model_name == 'Vcnet':
            init_lr = 0.0001
            alpha = 0.5

            Result['Vcnet'] = []

        elif model_name == 'Vcnet_tr':
            init_lr = 0.0001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.

            Result['Vcnet_tr'] = []

        for _ in range(num_dataset):

            cur_save_path = save_path + '/' + str(_)
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)

            data = pd.read_csv(load_path + '/' + str(_) + '/train.txt', header=None, sep=' ')
            train_matrix = torch.from_numpy(data.to_numpy()).float()
            data = pd.read_csv(load_path + '/' + str(_) + '/test.txt', header=None, sep=' ')
            test_matrix = torch.from_numpy(data.to_numpy()).float()
            data = pd.read_csv(load_path + '/' + str(_) + '/t_grid.txt', header=None, sep=' ')
            t_grid = torch.from_numpy(data.to_numpy()).float()

            # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
            train_loader = get_iter(train_matrix, batch_size=500, shuffle=True)
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

            # reinitialize model
            model._initialize_weights()

            # define optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

            if isTargetReg:
                TargetReg._initialize_weights()
                tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            print('model : ', model_name)
            for epoch in range(num_epoch):


                for idx, (inputs, y) in enumerate(train_loader):
                    t = inputs[:, 0]
                    x = inputs[:, 1:]

                    if isTargetReg:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                        loss.backward()
                        optimizer.step()

                        tr_optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        tr_loss = criterion_TR(out, trg, y, beta=beta)
                        tr_loss.backward()
                        tr_optimizer.step()
                    else:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        loss = criterion(out, y, alpha=alpha)
                        loss.backward()
                        optimizer.step()

                if epoch % verbose == 0:
                    print('current epoch: ', epoch)
                    print('loss: ', loss.data)

            if isTargetReg:
                t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)
            else:
                t_grid_hat, mse = curve(model, test_matrix, t_grid)

            mse = float(mse)
            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)
            print('-----------------------------------------------------------------')
            save_checkpoint({
                'model': model_name,
                'best_test_loss': mse,
                'model_state_dict': model.state_dict(),
                'TR_state_dict': TargetReg.state_dict() if isTargetReg else None,
            }, model_name=model_name, checkpoint_dir=cur_save_path)
            print('-----------------------------------------------------------------')

            Result[model_name].append(mse)

            with open(save_path + '/result.json', 'w') as fp:
                json.dump(Result, fp)


