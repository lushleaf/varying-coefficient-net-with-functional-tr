import torch

def x_t(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    t = (10. * torch.sin(max(x1, x2, x3)) + max(x3, x4, x5)**3)/(1. + (x1 + x5)**2) + \
        torch.sin(0.5 * x3) * (1. + torch.exp(x4 - 0.5 * x3)) + x3**2 + 2. * torch.sin(x4) + 2.*x5 - 6.5
    return t

def x_t_link(t):
    return 1. / (1. + torch.exp(-1. * t))

def t_x_y(t, x):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    y = torch.cos((t-0.5) * 3.14159 * 2.) * (t**2 + (4.*max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))
    return y

def simu_data1(n_train, n_test):
    train_matrix = torch.zeros(n_train, 8)
    test_matrix = torch.zeros(n_test, 8)
    for _ in range(n_train):
        x = torch.rand(6)
        train_matrix[_, 1:7] = x
        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)
        train_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, -1] = y

    for _ in range(n_test):
        x = torch.rand(6)
        test_matrix[_, 1:7] = x
        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)
        test_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        test_matrix[_, -1] = y

    t_grid = torch.zeros(2, n_test)
    t_grid[0, :] = test_matrix[:, 0].squeeze()

    for i in range(n_test):
        psi = 0
        t = t_grid[0, i]
        for j in range(n_test):
            x = test_matrix[j, 1:7]
            psi += t_x_y(t, x)
        psi /= n_test
        t_grid[1, i] = psi

    return train_matrix, test_matrix, t_grid




