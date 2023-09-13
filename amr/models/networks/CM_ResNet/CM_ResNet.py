import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import thop
__all__ = ["CM_ResNet", "test"]


class Residual_Stack(nn.Module):
    def __init__(self, in_channel, out_channel, kernelsize, poolsize, stride, padding):
        super(Residual_Stack, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1, bias = False)
        self.uint1 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=kernelsize, stride=stride, padding=padding, bias=False),
            nn.ReLU(out_channel),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        )
        self.uint2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=kernelsize, stride=stride, padding=padding, bias=False),
            nn.ReLU(out_channel),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(poolsize, stride=poolsize)

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.uint1(x1) + x1
        x3 = self.relu(x2)
        x4 = self.uint2(x3) + x3
        x5 = self.relu(x4)
        output = self.pool(x5)
        return output


class CM_ResNet(nn.Module):
    def __init__(self, classes):
        super(CM_ResNet, self).__init__()
        self.fc1 = nn.Linear(1024 * 2, 80)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(80, 2)

        self.ReStk0 = self.getLayer(2, 32, 3, 2, 1, 1)
        self.ReStk1 = self.getLayer(32, 32, 3, 2, 1, 1)
        self.ReStk2 = self.getLayer(32, 32, 3, 2, 1, 1)
        self.ReStk3 = self.getLayer(32, 32, 3, 2, 1, 1)
        self.ReStk4 = self.getLayer(32, 32, 3, 2, 1, 1)
        self.ReStk5 = self.getLayer(32, 32, 3, 2, 1, 1)

        self.flat = nn.Flatten()
        self.fc3 = nn.Sequential(nn.Linear(512, 128), nn.SELU(), nn.Dropout(0.3))
        self.fc4 = nn.Sequential(nn.Linear(128, 128), nn.SELU(), nn.Dropout(0.3))
        self.fc5 = nn.Sequential(nn.Linear(128, classes))
        self.init_weights()

    def init_weights(self):
        for l in self.modules():
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight.data)
                if l.bias is not None:
                    nn.init.constant_(l.bias.data, 0)
            if isinstance(l, nn.Conv1d):
                nn.init.xavier_uniform_(l.weight.data)
                if l.bias is not None:
                    nn.init.constant_(l.bias.data, 0)

    def getLayer(self, in_channel, out_channel, kernelsize, poolsize, stride, padding):
        layers = []
        layers.append(Residual_Stack(in_channel, out_channel, kernelsize, poolsize, stride, padding))
        return nn.Sequential(*layers)

    def forward(self, input):
        ####################  CM  start ############################
        x = input
        x1 = self.tanh(self.fc1(x.reshape(x.shape[0], -1)))
        x2 = self.fc2(x1)
        w = x2[:, 0].reshape(x.shape[0], -1)
        phi = x2[:, 1].reshape(x.shape[0], -1)

        n = torch.arange(0, 1024).reshape(1, -1).to(x.device).float()
        R = x[:, 0]
        I = x[:, 1]

        nR = R * torch.cos(w * n + phi) + I * torch.sin(w * n + phi)
        nI = I * torch.cos(w * n + phi) - R * torch.sin(w * n + phi)
        x3 = torch.stack((nR, nI), dim=1)

        input = x3
        ####################  CM  end ############################

        batchsize = input.shape[0]
        x = input.reshape([batchsize, 2, -1])
        x = self.ReStk0(x)
        x = self.ReStk1(x)
        x = self.ReStk2(x)
        x = self.ReStk3(x)
        x = self.ReStk4(x)
        x = self.ReStk5(x)

        x = self.flat(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


def test():
    net = CM_ResNet(25)
    batchsize = 1
    data_input = Variable(torch.randn([batchsize, 2, 1024]))
    flops, params = thop.profile(net, inputs=(data_input,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops & params", [flops, params])


if __name__ == '__main__':
    test()
