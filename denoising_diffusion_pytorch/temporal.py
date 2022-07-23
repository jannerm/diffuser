import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .mixer import *

class TemporalHelper(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size, final=False):
        super().__init__()
        padding = kernel_size // 2

        if final:
            self.block = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=padding)
        else:
            self.block = nn.Sequential(
                nn.Conv1d(inp_channels, out_channels, kernel_size, padding=padding),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(8, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                # nn.LayerNorm(out_channels),
                Mish(),
            )

    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size, embed_dim, horizon, final=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            TemporalHelper(inp_channels, out_channels, kernel_size),
            TemporalHelper(out_channels, out_channels, kernel_size, final=final),
        ])
        # self.mlp = nn.Sequential(
        #   nn.Linear(embed_dim, out_channels * 2),
        #   nn.Mish(),
        #   nn.Linear(out_channels * 2, out_channels),
        #   Rearrange('batch out -> batch out 1'),
        # )
        self.time_mlp = nn.Sequential(
            Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.cond_mlp = nn.Sequential(
            Mish(),
            nn.Linear(embed_dim, horizon),
            # Rearrange('batch h -> batch 1 h'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, cond):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

            returns:
                out : [ batch_size x out_channels x horizon ]
        '''
        # outer = torch.einsum('bi,bj->bij', self.time_mlp(t), self.cond_mlp(cond))
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TemporalDownsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class TemporalUpsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class TemporalMixerUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()
        # self.channels = channels

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                # MixerBlock(horizon, transition_dim, dim_in, dim_out, time_emb_dim=time_dim + cond_dim),
                # MixerBlock(horizon, transition_dim, dim_out, dim_out, time_emb_dim=time_dim + cond_dim),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                nn.Identity(),
                TemporalDownsample(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        # self.mid_block1 = MixerBlock(horizon, transition_dim, mid_dim, mid_dim, time_emb_dim=time_dim + cond_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        # self.mid_block2 = MixerBlock(horizon, transition_dim, mid_dim, mid_dim, time_emb_dim=time_dim + cond_dim)
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        # self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_attn = nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size=5, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                # MixerBlock(horizon, transition_dim, dim_out * 2, dim_in, time_emb_dim=time_dim + cond_dim),
                # MixerBlock(horizon, transition_dim, dim_in, dim_in, time_emb_dim=time_dim + cond_dim),
                # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                nn.Identity(),
                TemporalUpsample(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        # self.final_conv = nn.Sequential(
        #     Block(dim, dim),
        #     nn.Conv2d(dim, channels, 1),
        # )

        self.final_conv = nn.Sequential(
            TemporalHelper(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

        ## down
        ## [ 1, 32, 64, 128, 256 ]
        ## 0: B x 1 x 32 x 37 --> B x 32 x 16 x 37
        ## 1: --> B x 64 x 8 x 37
        ## 2: --> B x 128 x 4 x 37
        ## 3: --> B x 256 x 4 x 37

        ## block1: c x 2
        ## block2: ~
        ## attn  : ~
        ## down  : horizon / 2

        ## up
        ## 0: B x 256 x 4 x 37 --> B x 128 x 8 x 37
        ## 1: --> B x 64 x 16 x 37
        ## 2: --> B x 32 x 32 x 37

        ## cat: c x 2
        ## block1: c / 4
        ## block2: ~
        ## attn: ~
        ## up  : horizon * 2

        ## final
        ## B x 32 x 32 x 37 --> B x 1 x 32 x 37

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''
        cond = cond.to(x.device)
        t = self.time_mlp(time)
        cond = self.cond_mlp(cond)
        # cond = None

        h = []

        # x = x[:,None]
        t = torch.cat([t, cond], dim=-1)

        x = einops.rearrange(x, 'b h t -> b t h')

        for resnet, resnet2, attn, downsample in self.downs:
            # print('0', x.shape, t.shape)
            x = resnet(x, t, cond)
            # print('resnet', x.shape, t.shape)
            x = resnet2(x, t, cond)
            # print('resnet2', x.shape)
            ##
            x = einops.rearrange(x, 'b t h -> b t h 1')
            x = attn(x)
            x = einops.rearrange(x, 'b t h 1 -> b t h')
            ##
            # print('attn', x.shape)
            h.append(x)
            x = downsample(x)
            # print('downsample', x.shape, '\n')

        x = self.mid_block1(x, t, cond)
        ##
        x = einops.rearrange(x, 'b t h -> b t h 1')
        x = self.mid_attn(x)
        x = einops.rearrange(x, 'b t h 1 -> b t h')
        ##
        x = self.mid_block2(x, t, cond)
        # print('mid done!', x.shape, '\n')

        for resnet, resnet2, attn, upsample in self.ups:
            # print('0', x.shape)
            x = torch.cat((x, h.pop()), dim=1)
            # print('cat', x.shape)
            x = resnet(x, t, cond)
            # print('resnet', x.shape)
            x = resnet2(x, t, cond)
            # print('resnet2', x.shape)
            ##
            x = einops.rearrange(x, 'b t h -> b t h 1')
            x = attn(x)
            x = einops.rearrange(x, 'b t h 1 -> b t h')
            ##
            # print('attn', x.shape)
            x = upsample(x)
            # print('upsample', x.shape)
        # pdb.set_trace()
        x = self.final_conv(x)

        # x = x.squeeze(dim=1)

        ##
        x = einops.rearrange(x, 'b t h -> b h t')
        ##
        return x

class TemporalValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                nn.Identity(),
                TemporalDownsample(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = TemporalDownsample(mid_dim_2)
        horizon = horizon // 2
        ##
        # self.mid_attn = Residual(PreNorm(mid_dim_2, LinearAttention(mid_dim_2)))
        self.mid_attn = nn.Identity()
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = TemporalDownsample(mid_dim_3)
        horizon = horizon // 2

        fc_dim = mid_dim_3 * max(horizon, 1)
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def process_conditions(self, time):
        return self.time_mlp(time)

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''
        t = self.process_conditions(time, *args)
        cond = None

        h = []

        x = einops.rearrange(x, 'b h t -> b t h')

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t, cond)
            x = resnet2(x, t, cond)
            ##
            x = einops.rearrange(x, 'b t h -> b t h 1')
            x = attn(x)
            x = einops.rearrange(x, 'b t h 1 -> b t h')
            ##
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, cond)
        x = self.mid_down1(x)
        ##
        x = einops.rearrange(x, 'b t h -> b t h 1')
        x = self.mid_attn(x)
        x = einops.rearrange(x, 'b t h 1 -> b t h')
        ##
        x = self.mid_block2(x, t, cond)
        x = self.mid_down2(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out

class MetaValueFunction(TemporalValueFunction):

    def __init__(self, *args, dim=32, **kwargs):
        super().__init__(*args, dim=dim, time_dim=dim*2, **kwargs)

        ## overwrite `time_mlp` to have the original `dim` output

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(1, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )

    def process_conditions(self, time, value):
        return torch.cat([
            self.time_mlp(time),
            self.value_mlp(value),
        ], dim=-1)
