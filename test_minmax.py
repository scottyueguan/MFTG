import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class ValueNet(nn.Module):

    def __init__(self, input_size, output_size, layer_size_list):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_size_list[0]))
        for i in range(1, len(layer_size_list)):
            self.layers.append(nn.Linear(layer_size_list[i - 1], layer_size_list[i]))
        self.layers.append(nn.Linear(layer_size_list[-1], output_size))

    def forward(self, x):
        tmp = self.layers[0](x)
        for layer in self.layers[1:len(self.layers)]:
            tmp = F.sigmoid(tmp)
            tmp = layer(tmp)
        return tmp


def func(x, y):
    return x ** 2 - y ** 2


if __name__ == "__main__":

    # train value net
    # value_net = ValueNet(2, 1, layer_size_list=[8, 16, 8])
    # optimzer = torch.optim.Adam(value_net.parameters(), lr=1e-2)
    #
    # criterion = torch.nn.MSELoss(reduction='mean')
    #
    # for t in range(10000):
    #     x, y = np.random.uniform(low=-1.0, high=1.0), np.random.uniform(low=-1.0, high=1.0)
    #     z = func(x, y)
    #
    #     optimzer.zero_grad()
    #     z_pred = value_net(torch.tensor([x, y]))
    #     z = torch.tensor([z])
    #
    #     loss = criterion(z, z_pred)
    #     loss.backward()
    #     optimzer.step()
    # print(loss)

    # torch.save(value_net, "value_net.pt")

    # load value net
    value_net = torch.load("value_net.pt")

    # perform min-max
    x = torch.tensor([10.0, 15.0])
    x.requires_grad = True

    for _ in range(1000):
        # theta = torch.exp(x) / torch.sum(torch.exp(x))
        theta = x
        y = (theta[0] - 0.0) ** 2 - (theta[1] - 0.0) ** 2
        gradient = torch.autograd.grad(outputs=y, inputs=x, retain_graph=True)[0]
        da_coeff = torch.tensor([-1, 1])
        step = torch.mul(da_coeff, gradient)
        x = torch.add(x, 1e-2 * step)
        x = torch.clip(x, min=-20, max=20)

    result = x.cpu().detach().numpy()
    print(result)
    point = [result[0], result[1], result[0] ** 2 - result[1] ** 2]
    # print(np.exp(result) / np.sum(np.exp(result)))

    # visualize result
    X = np.arange(-1, 1, 0.01)
    Y = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = (X-0.0) ** 2 - (Y-0.0) ** 2

    input = np.column_stack((X.reshape((-1, 1)), Y.reshape((-1, 1))))
    z_pred = value_net(torch.tensor(input).to(torch.float32))
    Z_pred = z_pred.cpu().detach().numpy()
    Z_pred = Z_pred.reshape((200, 200))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(X, Y, Z)
    ax.plot_surface(X, Y, Z_pred)
    ax.plot(point[0], point[1], point[2], 'r.', markersize=20)

    plt.show()
