from functools import partial
import torch
import torch.nn as nn
import pdb

## @TODO : don't do this
# from ..denoising_diffusion_pytorch import *
from .helpers import *
from .mlp import MLP

def exists(x):
        return (x is not None)

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
        ) # if exists(time_emb_dim) else None


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

        if exists(self.mlp):
            # print('hmmm')
            h += self.mlp(time_emb)[:, :, None, None]

        return h + self.res_conv(x)


class MixerUnet(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        height, width = image_size

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            print("inputs: ", dim_in, dim_out, height, width)
            import pdb
            pdb.set_trace()
            print(height)
            print(width)

            self.downs.append(nn.ModuleList([
                MixerBlock(height, width, dim_in, dim_out, time_emb_dim = time_dim),
                MixerBlock(height, width, dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))


            if not is_last:
                height = height // 2

        mid_dim = dims[-1]
        self.mid_block1 = MixerBlock(height, width, mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = MixerBlock(height, width, mid_dim, mid_dim, time_emb_dim = time_dim)

        import pdb
        pdb.set_trace()
        print("here")

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                MixerBlock(height, width, dim_out * 2, dim_in, time_emb_dim = time_dim),
                MixerBlock(height, width, dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                height = height * 2

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )
    def forward(self, x, mask, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []
        x = torch.cat([x, mask], dim=1)

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
            # pdb.set_trace()

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

        return self.final_conv(x)

#--------------------------------- refactor ---------------------------------#

# from einops.layers.torch import Rearrange
from einops import rearrange

class SmallMixerBlock(nn.Module):

    def __init__(self, height, width, channels, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.ln1 = nn.LayerNorm(channels * width)
        self.block1 = FeedForward(channels * width, 4*channels)

        self.ln2 = nn.LayerNorm(height)
        self.block2 = FeedForward(height, 4*height)

        self.output_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()
        self.res_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()

        # self.output_mlp = nn.Sequential(
        #     nn.Linear(channels, dim_out * 4),
        #     nn.GELU(),
        #     nn.Linear(dim_out * 4, dim_out),

        # )
        # self.dim_out = dim_out

        self.output_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()
        self.res_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()

        # self.rearrange_1 = Rearrange('b c h w -> b h (c w)')
        # self.rearrange_2 = Rearrange('b c h w -> b (c w) h')

        # assert width % 2 == 1
        # padding = (width - 1) // 2
        # self.output_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, width), padding=(0, padding)) if dim != dim_out else nn.Identity()
        # self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, width), padding=(0, padding)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        '''
            x : [ batch_size x channels x horizon x joined_dim ]
        '''
        bs, ch, height, width = x.size()

        x_1 = rearrange(x, 'b c h w -> b h (c w)')

        x_1 = self.ln1(x_1)
        x_1 = self.block1(x_1)
        x_1 = rearrange(x_1, 'b h (c w) -> b c h w', c=ch)

        # x = x + x_1

        x_2 = rearrange(x, 'b c h w -> b (c w) h')
        x_2 = self.ln2(x_2)
        x_2 = self.block2(x_2)
        x_2 = rearrange(x_2, 'b (c w) h -> b c h w', c=ch)

        x = x_1 + x_2

        # x = rearrange(x, 'b c h w -> b w (c h)')
        # x = self.output_mlp(x)
        # x = rearrange(x, 'b w (c h) -> b c h w', c=self.dim_out)

        # x = rearrange(x, 'b c h w -> b h w c')
        # x = self.output_mlp(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x + self.mlp(time_emb)[:, :, None, None]


        x = self.output_conv(x)
        x = x + self.mlp(time_emb)[:, :, None, None]

        return x

class SmallMixerUnet(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        height, width = image_size

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                SmallMixerBlock(height, width, dim_in, dim_out, time_emb_dim = time_dim),
                SmallMixerBlock(height, width, dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                height = height // 2

        mid_dim = dims[-1]
        self.mid_block1 = SmallMixerBlock(height, width, mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = SmallMixerBlock(height, width, mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                SmallMixerBlock(height, width, dim_out * 2, dim_in, time_emb_dim = time_dim),
                SmallMixerBlock(height, width, dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                height = height * 2

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )
    def forward(self, x, mask, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []
        x = torch.cat([x, mask], dim=1)

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
            # pdb.set_trace()

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

        return self.final_conv(x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class TimeLinear(nn.Module):
    def __init__(self, inp_dim, out_dim, cond_dim):
        super().__init__()
        self.layer = nn.Linear(inp_dim + cond_dim, out_dim)

    def forward(self, x, *conds):
        '''
            x : [ batch_size x n_patches x inp_dim ]
            cond : [ batch_size x cond_dim ]
            returns : [ batch_size x n_patches x out_dim ]
        '''
        rep = x.shape[1]
        cond = torch.cat(conds, dim=-1)
        cond = cond.unsqueeze(1).repeat(1, rep, 1)
        joined = torch.cat([x, cond], dim=-1)
        return self.layer(joined)

class TimeMLP(nn.Module):

    def __init__(self, inp_dim, expansion_factor, cond_dim):
        self.fc1 = TimeLinear(inp_dim, inp_dim * expansion_factor, cond_dim)
        self.fc2 = TimeLinear(inp_dim * expansion_factor, inp_dim, cond_dim)

    def forward(self, x, time_embed):
        x = self.fc1(x, time_embed)
        x = F.gelu(x)
        x = self.fc2(x, time_embed)
        return x


def FeedForward(dim, expansion_factor=4, dropout=0., layer=nn.Linear):
    return nn.Sequential(
        layer(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        layer(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class PatchMixerBlock(nn.Module):

    def __init__(self, n_patches, patch_dim, cond_dim, expansion_factor=1):
        super().__init__()


        self.n_patches = n_patches
        # self.mlp = nn.Sequential(
        #     Mish(),
        #     nn.Linear(time_emb_dim, dim_out)
        # ) if exists(time_emb_dim) else None

        # self.ln1 = nn.LayerNorm(channels * width)
        # self.block1 = FeedForward(channels * width, 4*channels)

        # self.ln2 = nn.LayerNorm(height)
        # self.block2 = FeedForward(height, 4*height)

        # self.output_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()
        # self.res_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()

        # # self.output_mlp = nn.Sequential(
        # #     nn.Linear(channels, dim_out * 4),
        # #     nn.GELU(),
        # #     nn.Linear(dim_out * 4, dim_out),

        # # )
        # # self.dim_out = dim_out

        # self.output_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()
        # self.res_conv = nn.Conv2d(channels, dim_out, 1) if channels != dim_out else nn.Identity()

        # # self.rearrange_1 = Rearrange('b c h w -> b h (c w)')
        # # self.rearrange_2 = Rearrange('b c h w -> b (c w) h')

        # # assert width % 2 == 1
        # # padding = (width - 1) // 2
        # # self.output_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, width), padding=(0, padding)) if dim != dim_out else nn.Identity()
        # # self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, width), padding=(0, padding)) if dim != dim_out else nn.Identity()

        # layer = partial(TimeLinear, cond_dim=cond_dim)
        # self.fc2 = PreNormResidual(patch_dim, FeedForward(patch_dim, layer=layer))

    def forward(self, x, time_embed):
        '''
            x : [ batch_size x n_patches x patch_dim ]
            time_embed : [ batch_size x time_embed_dim ]
        '''

        # time_embed = time_embed.unsqueeze(1).repeat(1, self.n_patches, 1)

        # joined = torch.cat([x, time_embed], dim=-1)
        # self.fc2(joined)
        pdb.set_trace()


        # bs, height, width = x.size()

        # x_1 = rearrange(x, 'b h w -> b h w')

        # x_1 = self.ln1(x_1)
        # x_1 = self.block1(x_1)
        # x_1 = rearrange(x_1, 'b h (c w) -> b c h w', c=ch)

        # # x = x + x_1

        # x_2 = rearrange(x, 'b c h w -> b (c w) h')
        # x_2 = self.ln2(x_2)
        # x_2 = self.block2(x_2)
        # x_2 = rearrange(x_2, 'b (c w) h -> b c h w', c=ch)

        # x = x_1 + x_2

        # # x = rearrange(x, 'b c h w -> b w (c h)')
        # # x = self.output_mlp(x)
        # # x = rearrange(x, 'b w (c h) -> b c h w', c=self.dim_out)

        # # x = rearrange(x, 'b c h w -> b h w c')
        # # x = self.output_mlp(x)
        # # x = rearrange(x, 'b h w c -> b c h w')
        # # x = x + self.mlp(time_emb)[:, :, None, None]


        # x = self.output_conv(x)
        # x = x + self.mlp(time_emb)[:, :, None, None]

        # return x

class PatchMixer(nn.Module):

    def __init__(self, horizon, transition_dim, cond_dim, transition_embed_dim=64, dim_mults=(1, 2, 4, 8), time_embed_dim=32, *args, **kwargs):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )

        self.transition_embed_dim = transition_embed_dim
        self.time_embed_dim = time_embed_dim
        self.cond_dim = cond_dim

        self.horizon_patch_size = 1
        self.n_horizon_patches = horizon // self.horizon_patch_size
        self.transition_patch_size = transition_dim * self.horizon_patch_size

        self.fc1 = TimeLinear(self.transition_patch_size, self.transition_embed_dim, self.cond_dim + self.time_embed_dim)
        self.fc2 = TimeLinear(self.transition_embed_dim, self.transition_patch_size, self.cond_dim + self.time_embed_dim)

        self.block_mid = nn.Sequential(
            nn.LayerNorm(self.transition_embed_dim),
            nn.Conv1d(dim_mults[-1], dim_mults[-1], kernel_size=3, padding=1),
            nn.LayerNorm(self.transition_embed_dim),
            TimeLinear(self.transition_embed_dim, self.transition_embed_dim, self.cond_dim + self.time_embed_dim),
        )

        blocks_down = []
        blocks_up = []
        dim = 1
        for mult in dim_mults:
            block_down = nn.Sequential(
                nn.LayerNorm(self.transition_embed_dim),
                nn.Conv1d(dim, mult, kernel_size=3, padding=1),
                nn.LayerNorm(self.transition_embed_dim),
                TimeLinear(self.transition_embed_dim, self.transition_embed_dim, self.cond_dim + self.time_embed_dim),
            )
            block_up = nn.Sequential(
                nn.LayerNorm(self.transition_embed_dim),
                nn.Conv1d(2 * mult, dim, kernel_size=3, padding=1),
                nn.LayerNorm(self.transition_embed_dim),
                TimeLinear(self.transition_embed_dim, self.transition_embed_dim, self.cond_dim + self.time_embed_dim),
            )
            blocks_down.append(block_down)
            blocks_up.insert(0, block_up)
            dim = mult

        self.blocks_down = nn.ModuleList(blocks_down)
        self.blocks_up = nn.ModuleList(blocks_up)


        # self.block = PatchMixerBlock(self.n_horizon_patches, self.transition_embed_dim, self.time_embed_dim)
        # self.blah = nn.Linear(10, 10)

    def forward(self, x, cond, time):
        '''
            x : [ batch_size x 1 x horizon x joined_dim ]
            cond : [ batch_size x obs_dim ]
            time : [ batch_size ]
        '''

        ## [ batch_size x horizon x joined_dim ]
        x = x.squeeze(1)

        ## [ batch_size x 1 x dim ]
        time_embed = self.time_mlp(time)

        ## [ batch_size x n_horizon_patches x transition_embed_dim ]
        x = self.fc1(x, cond, time_embed)


        # pdb.set_trace()
        h = []

        for norm1, conv, norm2, fc in self.blocks_down:
            # print('0', x.shape)
            x = norm1(x)
            # print('1', x.shape)
            x = conv(x)
            # print('2', x.shape)
            x = norm2(x)
            # print('3', x.shape)
            x = fc(x, cond, time_embed)
            # print('4', x.shape)
            h.append(x)
        # pdb.set_trace()


        for norm1, conv, norm2, fc in [self.block_mid]:
            x = norm1(x)
            x = conv(x)
            x = norm2(x)
            x = fc(x, cond, time_embed)
        # pdb.set_trace()

        for norm1, conv, norm2, fc in self.blocks_up:
            # print('0', x.shape)
            x = torch.cat((x, h.pop()), dim=1)
            # print('0', x.shape)
            x = norm1(x)
            # print('1', x.shape)
            x = conv(x)
            # print('2', x.shape)
            x = norm2(x)
            x = fc(x, cond, time_embed)

        x = self.fc2(x, cond, time_embed)

        # assert x.shape[1] == 1
        # obs_dim = cond.shape[1]
        # x[:, 0, :obs_dim] = x[:, 0, :obs_dim] + cond
        # x[:, 0, :obs_dim] = cond

        x = x.unsqueeze(1)
        # pdb.set_trace()
        return x

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
