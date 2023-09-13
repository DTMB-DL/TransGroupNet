import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import thop
from einops import rearrange, repeat

__all__ = ["TransNet", "test"]


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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.swish = Swish()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        f1 = x1 * self.swish(x2)
        f = self.fc3(f1)
        return f


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., talk_heads=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.talking_heads1 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()
        self.talking_heads2 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = self.talking_heads1(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn = self.talking_heads2(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class GroupBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(GroupBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, in_channel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential(
            nn.Identity(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x3 = self.relu(x1 + x2)
        return x3

    def __call__(self, x):
        return self.forward(x)


class TransNet(nn.Module):
    def __init__(self, classes, dim=96):
        super(TransNet, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.We = nn.Parameter(torch.randn(1, dim, dim))
        self.dropout = nn.Dropout(0.1)
        self.preconv1 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=32, stride=16),
            nn.ReLU()
        )
        self.preconv2 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=32, stride=16),
            nn.ReLU()
        )
        self.preconv3 = nn.Sequential(
            nn.Conv1d(1, dim//3, kernel_size=32, stride=16),
            nn.ReLU()
        )
        self.coder = Transformer(dim=dim, depth=6, heads=8, dim_head=dim // 8, mlp_dim=dim * 2, dropout=0.3)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes)
        )

    def forward(self, x):
        x = x.float()
        A = self.preconv1(x[:,0].reshape(x.shape[0],1,-1)).transpose(1, 2)
        P = self.preconv2(x[:,1].reshape(x.shape[0],1,-1)).transpose(1, 2)
        F = self.preconv3(x[:,2].reshape(x.shape[0],1,-1)).transpose(1, 2)
        x = torch.cat((A,P,F), dim=2)
        B, N, _ = x.shape

        # class token & position coding
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=B)  # (B, 1, D)
        x = torch.cat((cls_tokens, torch.matmul(x, self.We)), dim=1)
        x += self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)

        codex = self.coder(x)[:, 0]
        fc = self.fc(codex.reshape(B, -1))

        return fc


def test():
    x = torch.randn(1, 3, 1024)
    net = TransNet(25)
    flops, params = thop.profile(net, inputs=(x,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops & params", [flops, params])


if __name__ == "__main__":
    test()
