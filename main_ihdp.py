import torch
import math
import numpy as np
from models.dynamic_net import Vcnet, Drnet, TR
from data.data import get_iter
from utils.eval import curve

import os
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

def save_checkpoint(state, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/ihdp', help='dir of data matrix')
    parser.add_argument('--data_split_dir', type=str, default='dataset/ihdp/eval/0', help='dir data split')
    parser.add_argument('--save_dir', type=str, default='logs/ihdp/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')

    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Parameters

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 1e-3

    # epoch: 800!
    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # get data
    data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt')

    idx_train = torch.load(args.data_split_dir + '/idx_train.pt')
    idx_test = torch.load(args.data_split_dir + '/idx_test.pt')

    train_matrix = data_matrix[idx_train, :]
    test_matrix = data_matrix[idx_test, :]
    t_grid = t_grid_all[:, idx_test]

    n_data = data_matrix.shape[0]

    # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
    train_loader = get_iter(data_matrix[idx_train,:], batch_size=471, shuffle=True)
    test_loader = get_iter(data_matrix[idx_test,:], batch_size=data_matrix[idx_test,:].shape[0], shuffle=False)

    grid = []
    MSE = []

    save_path = ''
    # for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr']:
    for model_name in ['Drnet_tr', 'Vcnet_tr']:
        # import model
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        # use Target Regularization?
        if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
            isTargetReg = 1
        else:
            isTargetReg = 0

        #if isTargetReg:
        tr_knots = list(np.arange(0.05, 1, 0.05))
        tr_degree = 2
        TargetReg = TR(tr_degree, tr_knots)
        TargetReg._initialize_weights()

        # best cfg for each model
        if model_name == 'Tarnet':
            init_lr = 0.02
            alpha = 1.0
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Tarnet_tr':
            init_lr = 0.02
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Drnet':
            init_lr = 0.02
            alpha = 1.
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Drnet_tr':
            init_lr = 0.02
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Vcnet':
            init_lr = 0.001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Vcnet_tr':
            # init_lr = 0.0001
            init_lr = 0.001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

        if isTargetReg:
            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

        print('model = ', model_name)
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
                print('loss: ', loss)

        if isTargetReg:
            t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)
            mse = float(mse)
            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)
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
            'TR_state_dict': TargetReg.state_dict(),
        }, checkpoint_dir=save_path)

        print('-----------------------------------------------------------------')

        grid.append(t_grid_hat)
        MSE.append(mse)


    import matplotlib.pyplot as plt
    # plot
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    font_legend = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    plt.figure(figsize=(5, 5))

    # b = 'C0'
    # o = 'C1'
    # g = 'C2'

    c1 = '#F87664'
    c1 = 'red'
    c1 = 'gold'
    c2 = '#00BA38'
    c2 = 'red'
    c3 = '#619CFF'
    c3 = 'dodgerblue'

    truth_grid = t_grid[:,t_grid[0,:].argsort()]
    x = truth_grid[0, :]
    y = truth_grid[1, :]
    plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)

    x = grid[1][0, :]
    y = grid[1][1, :]
    plt.scatter(x, y, marker='h', label='Vcnet', alpha=1, zorder=2, color=c2, s=20)

    x = grid[0][0, :]
    y = grid[0][1, :]
    plt.scatter(x, y, marker='H', label='Drnet', alpha=1, zorder=3, color=c3, s=20)

    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(-1.5, 3.0, 0.5), fontsize=0, family='Times New Roman')
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=0, family='Times New Roman')
    plt.grid()
    #plt.legend(prop=font_legend, loc='upper left')
    plt.xlabel('Treatment', font1)
    plt.ylabel('Response', font1)

    plt.savefig("Vc_Dr_ihdp.pdf", bbox_inches='tight')
    plt.show()
