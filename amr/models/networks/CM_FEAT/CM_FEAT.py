import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import thop
from einops import rearrange, repeat

__all__ = ["CM_FEAT", "test"]


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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForward2(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        f1 = x1*self.gelu(x2)
        f2 = x2*self.gelu(x1)
        f = self.fc3(f1+f2)
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
        qkv = self.to_qkv(x).chunk(3, dim = -1)
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
                PreNorm(dim, FeedForward2(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CM_FEAT(nn.Module):
    def __init__(self, classes, dim=64):
        super(CM_FEAT, self).__init__()
        self.fc1 = nn.Linear(1024 * 2, 80)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(80, 2)

        self.pos_embedding = nn.Parameter(torch.randn(1, 64, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.We = nn.Parameter(torch.randn(1, dim, dim))
        self.dropout = nn.Dropout(0.1)
        self.coder = Transformer(dim=dim, depth=6, heads=8, dim_head=dim//8, mlp_dim=dim * 2, dropout=0.3)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes)
        )

    def forward(self, x):
        ####################  CM  start ############################
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

        x = x3
        ####################  CM  end ############################

        t_max = torch.max(x)
        t_min = torch.min(x)
        diff = t_max - t_min
        t = (x - t_min) / diff

        # [B,2, 1024]
        Y = []
        for idx in range(0, t.shape[-1] - 32 + 1, 16):
            Y.append(t[:, :, idx:idx + 32].reshape(t.shape[0], -1))  # (2, L=32)
        x = torch.stack(Y, dim=1)  # (F, 2L) = (63, 64)  F=(1024-L)/R+1

        # B, F, 3L  =  B, 63, 96
        B, N, _ = x.shape
        x = x.float()

        # class token & position coding
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=B)  # (B, 1, D)
        x = torch.cat((cls_tokens, torch.matmul(x, self.We)), dim=1)
        x += self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)  # 此处dropout有效缓解后期过拟合问题

        # B, F+1, 3L  =  B, 64, 96
        codex = self.coder(x)[:, 0]
        fc = self.fc(codex.reshape(B, -1))

        return fc


def test():
    # x = torch.randn(1, 63, 64)
    x = torch.randn(1, 2, 1024)
    net = CM_FEAT(25)
    flops, params = thop.profile(net, inputs=(x,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops & params", [flops, params])


if __name__ == "__main__":
    test()
