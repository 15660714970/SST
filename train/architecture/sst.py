import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np
from thop import profile

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class NPM(nn.Module):
    def __init__(self, in_channel):
        super(NPM, self).__init__()
        self.in_channel = in_channel
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv0_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv0_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_0_cat = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)

        self.conv2_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv2_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_2_cat = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)

        self.conv4_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv4_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_4_cat = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)

        self.conv_cat = nn.Conv2d(in_channel * 3, in_channel, 3, 1, 1)  # 修改 3

    def forward(self, x):
        x_0 = x
        x_2 = F.avg_pool2d(x, 2, 2)
        x_4 = F.avg_pool2d(x_2, 2, 2)

        x_0 = torch.cat([self.conv0_33(x_0), self.conv0_11(x_0)], 1)
        x_0 = self.activation(self.conv_0_cat(x_0))

        x_2 = torch.cat([self.conv2_33(x_2), self.conv2_11(x_2)], 1)
        x_2 = F.interpolate(self.activation(self.conv_2_cat(x_2)), scale_factor=2, mode='bilinear',
                            align_corners=False)

        x_4 = torch.cat([self.conv2_33(x_4), self.conv2_11(x_4)], 1)
        x_4 = F.interpolate(self.activation(self.conv_4_cat(x_4)), scale_factor=4, mode='bilinear',
                            align_corners=False)
        x = x + self.activation(self.conv_cat(torch.cat([x_0, x_2, x_4], 1)))  # 修改

        return x

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            heads,
    ):
        super().__init__()
        self.num_heads = heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim)
        self.proj_p = nn.Linear(dim, dim)
        self.pos_emb1 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1, bias=False, groups=dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1, bias=True, groups=dim // 2),
        )
        self.pos_emb2 = nn.Sequential(
            nn.Conv2d(dim - dim // 2, dim - dim // 2, 3, 1, 3, dilation=3, bias=False, groups=dim - dim // 2),
            nn.GELU(),
            nn.Conv2d(dim - dim // 2, dim - dim // 2, 3, 1, 3, dilation=3, bias=True, groups=dim - dim // 2),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        x_list = []
        # 每个头的基础通道数
        base_channels_per_head = self.dim // self.num_heads
        # 计算剩余的通道数
        remaining_channels = self.dim % self.num_heads
        # 分配通道
        channels_per_head = [base_channels_per_head] * (self.num_heads)
        for i in range(remaining_channels):
            channels_per_head[i] += 1

        for j in range(self.num_heads):
            indice = 0
            q, k, v = [q_inp[:,:,indice:indice+channels_per_head[j]], k_inp[:,:,indice:indice+channels_per_head[j]],
                       v_inp[:, :, indice:indice + channels_per_head[j]]]
            indice += channels_per_head[j]

            # q: b,hw,c
            q = q.transpose(-2, -1)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            q = F.normalize(q, dim=-1, p=2)
            k = F.normalize(k, dim=-1, p=2)
            attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
            attn = attn / math.sqrt(channels_per_head[j]) * self.rescale[j,:,:]
            attn = attn.softmax(dim=-1)
            x = attn @ v  # b,c,hw
            x = x.permute(0, 2, 1)  # Transpose
            x_list.append(x)
        x = torch.cat(x_list, dim=-1)
        out_c = self.proj(x).view(b, h, w, c)

        v1 = v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)
        v1_1 = v1[:, :c // 2, :, :]
        v1_2 = v1[:, c // 2:, :, :]
        out_p1 = self.pos_emb1(v1_1).permute(0, 2, 3, 1)
        out_p2 = self.pos_emb2(v1_2).permute(0, 2, 3, 1)
        out_p = torch.cat([out_p1,out_p2], dim=-1).contiguous()
        out_p = self.proj_p(out_p)
        out = out_c + out_p

        return out

class DWConv(nn.Module):
    def __init__(self, dim=31):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        hidden_features = int(2 * hidden_features)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b,h,w,c = x.shape
        x = x.view(b, -1, c)
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, h, w)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(b, h, w, c).contiguous()
        return x

class FeedForward1(nn.Module):
    def __init__(self, dim, mult=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*mult)
        self.act = nn.GELU()
        self.dwconv = DWConv(dim*mult)
        self.fc2 = nn.Linear(dim*mult, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        x = x.view(b, -1, c)
        x = self.fc1(x)
        x = self.act(self.dwconv(x, h, w) + x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(b, h, w, c).contiguous()
        return x


class FeedForward(nn.Module):
    def __init__(self, mult=4):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv3d(1, mult, 1, 1, 0, bias=True),
                GELU(),
                nn.Conv3d(mult, mult, 3, 1, 1, bias=True, groups=mult),
                GELU(),
                nn.Conv3d(mult, 1, 1, 1, 0, bias=True),
            )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.unsqueeze(1))
        return out.squeeze(1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, heads=heads),
                PreNorm(dim, ConvolutionalGLU(dim)),
                PreNorm(dim, FeedForward(mult=1))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, glu, ff) in self.blocks:
            x = attn(x) + x
            x = glu(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class SSU(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4]):
        super(SSU, self).__init__()
        self.dim = dim
        self.stage = stage
        self.norm = nn.LayerNorm(dim)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        head = 1
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(dim=dim, num_blocks=num_blocks[i], heads=head),
                nn.Conv2d(dim, dim, 4, 2, 1, bias=False),
            ]))
            head *= 2   #2
        # Bottleneck
        self.bottleneck = MSAB(dim=dim, heads=head, num_blocks=num_blocks[-1])

        head //= 2   #2

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(2 * dim, dim, 1, 1, bias=False),
                MSAB(dim=dim, num_blocks=num_blocks[stage - 1 - i], heads=head),
            ]))
            head //= 2  #2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out

class SST(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(SST, self).__init__()
        self.stage = stage
        self.denosing = NPM(in_channels)
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1, bias=False)
        modules_body = [SSU(dim=31, stage=3, num_blocks=[1, 1, 1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.denosing(x)
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]

if __name__ == "__main__":
    input_tensor = torch.rand(1, 3, 256, 256)
    model = SST(3, 31, 31, 3)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    print(output_tensor.size())

    gmac, param = profile(model, (input_tensor, ))
    print(f'GMac:{gmac / (1024 * 1024 * 1024)}')
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print(torch.__version__)

