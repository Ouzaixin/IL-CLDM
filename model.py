import math
import torch
from torch import nn
import config
from inspect import isfunction

# Basic components

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)

# Adversarial Autoencoder (AAE)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim =config.latent_dim):
        super(Encoder, self).__init__()
        channels = [16, 32, 64]
        num_res_blocks = 1
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != len(channels)-1:
                layers.append(Downsample(channels[i+1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)
       
    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim = config.latent_dim):
        super(Decoder, self).__init__()
        channels = [64, 32, 16]
        num_res_blocks = 1

        in_channels = channels[0]
        layers = [nn.Conv3d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != 0:
                layers.append(Upsample(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AAE(nn.Module):
    def __init__(self):
        super(AAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data):
        encoded_data = self.encoder(data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

class Discriminator(nn.Module):
    def __init__(self, image_channels = 1, channels = [16, 32, 64, 128]):
        super(Discriminator, self).__init__()

        layers = [nn.Conv3d(image_channels, channels[0], 4, 2, 1), nn.LeakyReLU(0.2)]
        layers += [
            nn.Conv3d(channels[0], channels[1], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[1]),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [
            nn.Conv3d(channels[1], channels[2], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[2]),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [
            nn.Conv3d(channels[2], channels[3], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[3]),
            nn.LeakyReLU(0.2, True)
        ]

        layers.append(nn.Conv3d(channels[3], image_channels, 4, 2, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Unet

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, depth = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width, depth)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchwd, bncyxz -> bnhwdyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, depth, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, depth, height, width, depth)

        out = torch.einsum("bnhwdyxz, bncyxz -> bnchwd", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width, depth))

        return out + input

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=128,
        norm_groups=32,
        channel_mults=(1, 2, 2, 4),
        attn_res=(4,),
        res_blocks=2,
        dropout=0,
        with_time_emb=True,
        image_size=40
    ):
        super().__init__()

        if with_time_emb:
            time_dim = config.time_dim
            self.time_mlp = nn.Sequential(
                TimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                Swish(),
                nn.Linear(time_dim * 4, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups, 
                                dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        self.cond = nn.Sequential(
            nn.Conv3d(1, 16, 4, 2, 1),
            ResidualBlock(16, 16),
            GroupNorm(16),
            Swish(),
            nn.Conv3d(16, 1, 4, 2, 1),
            )
        self.label_emb = nn.Embedding(config.num_classes, time_dim)
    
    def forward(self, x, y, time, label=None):
        cond = self.cond(y)
        x =torch.cat([x,cond],dim=1)
        t = self.time_mlp(time)

        if label is not None:
            label = label.squeeze()
            label = label.long()
            k = self.label_emb(label)
            t =  t + k

    def forward(self, x, y, time, label=None):
        cond = self.condition(y)
        x = torch.cat([x,cond],dim=1)
        feats = []

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if label is not None:
            label = label.squeeze()
            label = label.long()
            k = self.label_emb(label)
            t =  t + k

        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
