import torch.nn as nn
from torch.autograd import Variable
import torch
import thop
import math
__all__ = ["CM_DAELSTM", "test"]
'''
reference: https://github.com/WuLoli/LSTMDAE
'''


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


# LSTM Auto-Encoder Class
class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio):
        super(LSTMAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio

        self.autoencoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout_ratio, batch_first=True)
        self.fc = TimeDistributed(nn.Linear(hidden_size, input_size), True)

    def forward(self, x):
        dec_out, (hidden_state, cell_state) = self.autoencoder(x)
        dec_out = self.fc(dec_out)
        return dec_out, hidden_state[-1]


class CM_DAELSTM(nn.Module):
    def __init__(self, n_classes, input_size=2, hidden_size=32, dropout_ratio=0.):
        super(CM_DAELSTM, self).__init__()
        self.fc1 = nn.Linear(1024 * 2, 80)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(80, 2)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.n_classes = n_classes
        self.lstmae = LSTMAE(input_size=input_size, hidden_size=hidden_size, dropout_ratio=dropout_ratio)
        self.clf_dense_1 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True),  # 激活函数
            nn.BatchNorm1d(32),  # 批量归一化,
            nn.Dropout(dropout_ratio)
        )
        self.clf_dense_2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(inplace=True),  # 激活函数
            nn.BatchNorm1d(16),  # 批量归一化,
            nn.Dropout(dropout_ratio)
        )
        self.clf_dense_3 = nn.Sequential(
            nn.Linear(in_features=16, out_features=n_classes),
        )

    def forward(self, x):
        ####################  CM  start ############################
        x1 = self.tanh(self.fc1(x.reshape(x.shape[0], -1)))
        x2 = self.fc2(x1)
        w = x2[:, 0].reshape(x.shape[0], -1)
        phi = x2[:, 1].reshape(x.shape[0], -1)

        n = torch.arange(0, 1024).reshape(1, -1).to(x.device).float()
        A = x[:, 0]
        P = x[:, 1]

        nA = A
        nP = P - (w * n - phi) / math.pi
        nP = (nP % 2) - 1
        x3 = torch.stack((nA, nP), dim=1)

        x = x3
        ####################  CM  end ############################

        rec_out, last_h = self.lstmae(x.transpose(1, 2))
        out_1 = self.clf_dense_1(last_h.reshape(x.shape[0],-1))
        out_2 = self.clf_dense_2(out_1)
        out_3 = self.clf_dense_3(out_2)
        return rec_out.transpose(1, 2), out_3


def test():
    net = CM_DAELSTM(25)
    batchsize = 1
    data_input = torch.randn([batchsize, 2, 1024])
    flops, params = thop.profile(net, inputs=(data_input,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops & params", [flops, params])


if __name__ == '__main__':
    test()