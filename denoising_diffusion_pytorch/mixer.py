from functools import partial
import torch
import torch.nn as nn
import einops
import pdb

## @TODO : don't do this
from .helpers import *
from .mlp import MLP

class DilateBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1)),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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


class MixerBlock(nn.Module):
    # Utilize a Mixer Block
    def __init__(self, height, width, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.ln1 = nn.LayerNorm(dim * width)
        self.block1 = FeedForward(dim * width, 4*dim)

        self.ln2 = nn.LayerNorm(height)
        self.block2 = FeedForward(height, 4*height)

        self.output_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # assert width % 2 == 1
        # padding = (width - 1) // 2
        # self.output_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, width), padding=(0, padding)) if dim != dim_out else nn.Identity()
        # self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, width), padding=(0, padding)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        '''
            x : [ batch_size x channels x horizon x joined_dim ]
        '''
        bs, ch, height, width = x.size()

        ## [ batch_size x horizon x (channels * joined_dim) ]
        x_first = x.permute(0, 2, 1, 3).reshape(bs, height, ch * width)

        # print(x_first.shape, self.ln1)
        x_first = self.ln1(x_first)
        x_first = self.block1(x_first)
        x_first = x_first.view(bs, height, ch, width).permute(0, 2, 1, 3)
        # x_first is of size N x C x 512 x W

        h = x_first

        ## [ batch_size x channels x joined_dim x horizon ]
        x_second = x.permute(0, 1, 3, 2).contiguous()
        x_second = self.ln2(x_second)
        x_second = self.block2(x_second).permute(0, 1, 3, 2).contiguous()

        h = h + x_second

        h = self.output_conv(h)
        h += self.mlp(time_emb)[:, :, None, None]

        return h + self.res_conv(x)


class MixerUnet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=1,
    ):
        super().__init__()
        self.channels = channels

        cond_dim = 0

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        transition_dim = transition_dim + 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                MixerBlock(horizon, transition_dim, dim_in, dim_out, time_emb_dim=time_dim + cond_dim),
                MixerBlock(horizon, transition_dim, dim_out, dim_out, time_emb_dim=time_dim + cond_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = MixerBlock(horizon, transition_dim, mid_dim, mid_dim, time_emb_dim=time_dim + cond_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = MixerBlock(horizon, transition_dim, mid_dim, mid_dim, time_emb_dim=time_dim + cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                MixerBlock(horizon, transition_dim, dim_out * 2, dim_in, time_emb_dim=time_dim + cond_dim),
                MixerBlock(horizon, transition_dim, dim_in, dim_in, time_emb_dim=time_dim + cond_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, channels, 1)
        )
    def forward(self, x, cond, time):
        t = self.time_mlp(time)

        h = []

        # x = x[:,None]
        x = torch.cat([x, cond[:, :, None].float().cuda()], dim=-1)
        x = x[:, None]
        # x = einops.rearrange(x, 'b c h t -> b t h c')
        # import pdb
        # pdb.set_trace()
        # print(x.size())

        # import pdb
        # pdb.set_trace()
        # print(t.size())
        # print(cond.size())
        # t = torch.cat([t, cond], dim=-1)

        for resnet, resnet2, attn, downsample in self.downs:
            # print('0', x.shape, t.shape)
            x = resnet(x, t)
            # print('resnet', x.shape, t.shape)
            x = resnet2(x, t)
            # print('resnet2', x.shape)
            x = attn(x)
            # print('attn', x.shape)
            h.append(x)
            x = downsample(x)
            # print('downsample', x.shape, '\n')

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # print('mid done!', x.shape, '\n')
        # pdb.set_trace()

        for resnet, resnet2, attn, upsample in self.ups:
            # print('0', x.shape)
            x = torch.cat((x, h.pop()), dim=1)
            # print('cat', x.shape)
            x = resnet(x, t)
            # print('resnet', x.shape)
            x = resnet2(x, t)
            # print('resnet2', x.shape)
            x = attn(x)
            # print('attn', x.shape)
            x = upsample(x)
            # print('upsample', x.shape)

        x = self.final_conv(x)

        # x = x.squeeze(dim=1)
        x = einops.rearrange(x, 'b 1 t h -> b t h')[..., :-1]
        return x

class ValueFunction(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        out_dim=1,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=1,
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                MixerBlock(horizon, transition_dim, dim_in, dim_out, time_emb_dim=time_dim + cond_dim),
                MixerBlock(horizon, transition_dim, dim_out, dim_out, time_emb_dim=time_dim + cond_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16
        ##
        self.mid_block1 = MixerBlock(horizon, transition_dim, mid_dim, mid_dim_2, time_emb_dim=time_dim + cond_dim)
        self.mid_down1 = Downsample(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_attn = Residual(PreNorm(mid_dim_2, LinearAttention(mid_dim_2)))
        # self.mid_down2 = Downsample(mid_dim_2)
        ##
        self.mid_block2 = MixerBlock(horizon, transition_dim, mid_dim_2, mid_dim_3, time_emb_dim=time_dim + cond_dim)
        ## [ B x mid_dim_3 x (height / 32) x width ]
        self.mid_down2 = Downsample(mid_dim_3)
        horizon = horizon // 2
        ##

        fc_dim = mid_dim_3 * max(horizon, 1) * transition_dim
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim, fc_dim // 8),
            Mish(),
            nn.Linear(fc_dim // 8, out_dim),
        )

    def forward(self, x, cond, time):
        t = self.time_mlp(time)

        x = x[:,None]
        t = torch.cat([t, cond], dim=-1)

        for resnet, resnet2, attn, downsample in self.downs:
            # print('0', x.shape, t.shape)
            x = resnet(x, t)
            # print('resnet', x.shape, t.shape)
            x = resnet2(x, t)
            # print('resnet2', x.shape)
            x = attn(x)
            # print('attn', x.shape)
            x = downsample(x)
            # print('downsample', x.shape, '\n')

        # print(x.shape)
        x = self.mid_block1(x, t)
        # print(x.shape)
        x = self.mid_down1(x)
        ##
        # print(x.shape)
        x = self.mid_attn(x)
        # print(x.shape)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        # print(x.shape, x.numel())
        x = x.view(len(x), -1)
        out = self.final_block(x)
        return out

class ImplicitValueFunction(MLP):

    def __init__(self, input_dim, *args, n_diffusion_steps=1000, embedding_dim=32, **kwargs):
        input_dim += embedding_dim
        super().__init__(input_dim, *args, **kwargs)
        self.embed = nn.Embedding(n_diffusion_steps, embedding_dim)

    def forward(self, x, cond, t):
        horizon = x.shape[1]
        assert horizon == 1
        x = x.squeeze(dim=1)

        t_embed = self.embed(t)
        joined = torch.cat([x, t_embed], dim=-1)
        return super().forward(joined)

#--------------------------------- refactor ---------------------------------#

class DebugMLP(nn.Module):

    def __init__(self, transition_dim, cond_dim, time_embed_dim=32, hidden_dim=256, pdrop=0.0, **kwargs):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )

        self.policy = nn.Sequential(
            nn.Linear(transition_dim + cond_dim + time_embed_dim, hidden_dim),
            nn.Dropout(pdrop),
            nn.GELU(),
            ##
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(pdrop),
            nn.GELU(),
            ##
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(pdrop),
            nn.GELU(),
            ##
            nn.Linear(hidden_dim, transition_dim),
            # nn.Tanh(),
        )

    def forward(self, x, observation, time):
        '''
            x : [ batch_size x horizon x transition_dim ]
            cond : [ batch_size x observation_dim ]
            time : [ batch_size x 1 ]
            returns : [ batch_size x 1 x action_dim ]
        '''

        ## [ batch_size x time_embed_dim ]
        time_embed = self.time_mlp(time)

        joined = torch.cat([x.squeeze(1), observation, time_embed], dim=-1)

        return self.policy(joined).unsqueeze(1)

    def loss(self, x, cond, time):

        pred = self.forward(cond, time)

        criterion = nn.MSELoss()
        loss = criterion(pred, x)

        return loss
