import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import pdb

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import imageio
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import diffusion.utils as utils

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, (4, 3), (2, 1), (1, 1))

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, (2, 1), 1)

    def forward(self, x):
        return self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        if exists(self.mlp):
            # print('hmmm')
            h += self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
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
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

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

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # def q_mean_variance(self, x_start, t):
    #     mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    #     variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
    #     log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    #     return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, mask, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, mask, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, mask, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, mask=mask, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        idx = 1

        sigma = (0.5 * model_log_variance).exp()

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * sigma * noise

    @torch.no_grad()
    def p_sample_loop(self, x, mask):
        device = self.betas.device

        shape = mask.shape
        b = shape[0]
        img = torch.randn(shape, device=device)
        img[mask.bool()] = x[mask.bool()].clone()

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, mask, torch.full((b,), i, device=device, dtype=torch.long))
            img[mask.bool()] = x[mask.bool()].clone()
        return img

    @torch.no_grad()
    def sample(self, x, mask):
        # image_size = self.image_size
        # channels = self.channels
        return self.p_sample_loop(x, mask)

    @torch.no_grad()
    def conditional_sample(self, batch_size, conditions):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        channels = 1 #self.channels
        shape = (batch_size, channels, *self.image_size)
        x = torch.zeros(*shape, device=device)
        mask = torch.zeros(*shape, device=device)
        for t, obs_dim, cond in conditions:
            x[:,:,t,:obs_dim] = cond
            mask[:,:,t,:obs_dim] = 1
        return self.p_sample_loop(x, mask)

    #### guided sampling

    @torch.no_grad()
    def guided_p_sample(self, guide, x, mask, t, clip_denoised=True, repeat_noise=False, cond_idx=0, stack_idx=0, place_idx=0):
        b, *_, device = *x.shape, x.device

        with torch.enable_grad():
            x.requires_grad_()
            y = guide(x, t)

            stack_xy = x[..., 64:, 7+stack_idx*8:9+stack_idx*8]
            place_xy = x[..., 64:, 7+place_idx*8:9+place_idx*8]

            dist = -0.1 * torch.abs(stack_xy - place_xy).mean(dim=-1).mean(dim=-1)

            grad = torch.autograd.grad([y[..., 64:, cond_idx].sum() + dist.sum()], [x])[0]
            # print(torch.abs(grad).mean())
            # grad_zero = torch.zeros_like(grad)

            min_idx = 7 + stack_idx * 8
            max_idx = 15 + stack_idx * 8

            for stack_idx_i in range(4):
                if stack_idx_i == stack_idx:
                    # grad[..., 14+stack_idx_i*8:15+stack_idx_i*8] = 0.05
                    continue

                grad[..., 7+stack_idx_i*8:14+stack_idx_i*8] = 0.0
                grad[..., 14+stack_idx_i*8:15+stack_idx_i*8] = -0.08
            # grad = torch.cat([grad[..., :7], grad_zero[..., 7:min_idx], grad[..., min_idx:max_idx], grad_zero[..., max_idx:]], dim=-1)
            x.detach()

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, mask=mask, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        sigma = (0.5 * model_log_variance).exp()
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        scalar = 0.7
        return model_mean + sigma * noise + sigma * grad * scalar, y

    @torch.no_grad()
    def guided_p_sample_loop(self, guide, x, mask, cond_idx, stack_idx, place_idx):
        device = self.betas.device

        shape = x.shape
        b = shape[0]
        img = torch.randn(shape, device=device)
        img[mask.bool()] = x[mask.bool()].clone()

        # progress = utils.Progress(self.num_timesteps)
        for i in tqdm(reversed(range(0, self.num_timesteps))):
            img, values = self.guided_p_sample(guide, img, mask, torch.full((b,), i, device=device, dtype=torch.long), cond_idx=cond_idx, stack_idx=stack_idx, place_idx=place_idx)
            # img = self.p_sample(img, mask, torch.full((b,), i, device=device, dtype=torch.long))
            img[mask.bool()] = x[mask.bool()].clone()

            # progress.update({
            #     't': i,
            #     'vmin': values.min().item(),
            #     'vmax': values.max().item(),
            # })
        # progress.stamp()

        ## sort by descending estimated values
        # inds = torch.argsort(values, descending=True)
        # img = img[inds]
        ##
        return img

    @torch.no_grad()
    def guided_conditional_sample(self, guide, batch_size, conditions, cond_idx, stack_idx, place_idx):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        channels = 1 #self.channels
        shape = (batch_size, *self.image_size)
        x = torch.zeros(*shape, device=device)
        mask  = torch.zeros(*shape, device=device)[..., 0]
        for t, cond, val in conditions:
            x[:,t,:] = val
            mask[:, t] = 1
        return self.guided_p_sample_loop(guide, x, mask, cond_idx, stack_idx, place_idx)

    @torch.no_grad()
    def fast_guided_conditional_sample(self, guide, batch_size, conditions, samples):
        '''
            conditions : [ (time, state), ... ]
        '''
        samples[:, :, :-1] = samples[:, :, 1:]
        # samples[:, :, -1:] = torch.randn_like(samples[:, :, -1:])

        # Do last 10 steps of sampling
        timestep = 30
        ts = torch.Tensor(np.ones(batch_size) * timestep).long()

        device = self.betas.device
        ts = ts.to(device)
        channels = 1
        shape = (batch_size, channels, *self.image_size)
        mask = torch.zeros(*shape, device=device)
        x = self.q_sample(samples, mask, ts)

        # x[:, :, :-1] = x[:, :, 1:]

        for t, obs_dim, cond in conditions:
            x[:,:,t,:obs_dim] = cond
            mask[:,:,t,:obs_dim] = 1

        shape = mask.shape
        img = x
        progress = utils.Progress(self.num_timesteps)

        b = shape[0]
        for i in reversed(range(0, timestep+1)):
            img, values = self.guided_p_sample(guide, img, mask, torch.full((b,), i, device=device, dtype=torch.long))
            img[mask.bool()] = x[mask.bool()].clone()

            progress.update({
                't': i,
                'vmin': values.min().item(),
                'vmax': values.max().item(),
            })
        progress.stamp()

        inds = torch.argsort(values, descending=True)
        img = img[inds]

        return img

    # #### conditional sampling
    # @torch.no_grad()
    # def p_conditional_sample_loop(self, shape, conditions):
    #     '''
    #         conditions : []
    #     '''
    #     device = self.betas.device

    #     b = shape[0]
    #     img = torch.randn(shape, device=device)
    #     img = self.enforce_conditions(img, conditions)

    #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
    #         img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
    #         img = self.enforce_conditions(img, conditions)
    #     return img

    # @torch.no_grad()
    # def enforce_conditions(self, x, conditions):
    #     '''
    #         x : [ B x 1 x H x obs_dim ]
    #     '''
    #     for t, cond in conditions:
    #         length = len(cond)
    #         x[:,:,t:t+length] = cond
    #     return x

    # @torch.no_grad()
    # def conditional_sample(self, batch_size = 16, conditions=[]):
    #     image_size = self.image_size
    #     channels = self.channels
    #     return self.p_conditional_sample_loop((batch_size, channels, *image_size), conditions)
    ####

    # @torch.no_grad()
    # def interpolate(self, x1, x2, t = None, lam = 0.5):
    #     b, *_, device = *x1.shape, x1.device
    #     t = default(t, self.num_timesteps - 1)

    #     assert x1.shape == x2.shape

    #     t_batched = torch.stack([torch.tensor(t, device=device)] * b)
    #     xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

    #     img = (1 - lam) * xt1 + lam * xt2
    #     for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
    #         img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

    #     return img

    def q_sample(self, x_start, mask, t, noise=None):
        noise = default(noise, lambda: conditional_noise(mask))

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        sample[mask.bool()] = x_start[mask.bool()]
        return sample

    def p_losses(self, x_start, mask, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: conditional_noise(mask))

        x_noisy = self.q_sample(x_start=x_start, mask=mask, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, mask, t)

        assert noise.shape == x_recon.shape
        if self.loss_type == 'l1':
            # loss = (noise - x_recon).abs().mean()
            weights = torch.tensor([1] * self.obs_dim + [self.act_weight] * self.act_dim, device=x_start.device, dtype=x_start.dtype)
            m = ~mask.bool()
            loss = (noise - x_recon).abs() * weights
            loss = loss[m].mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def set_weights(self, obs_dim, act_dim, act_weight = 1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_weight = act_weight

    def forward(self, x, mask, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, mask, t, *args, **kwargs)

def conditional_noise(mask):
    noise = torch.randn_like(mask)
    noise[mask.bool()] = 0
    return noise

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        renderer,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results'
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if type(folder) == str:
            self.ds = Dataset(folder, image_size)
        else:
            self.ds = folder
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.renderer = renderer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        milestone = (milestone // 50) * 50
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data, mask = next(self.dl)
                data = data.cuda()
                mask = mask.cuda()
                loss = self.model(data, mask)
                print(f'{self.step}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step == 0:
                inds = np.random.randint(0, len(self.ds), size=8)
                all_images_list = [self.ds[i][0] for i in inds]
                ## [ -1, 1 ]
                all_images = torch.cat(all_images_list, dim=0)
               ## [ 0, 1 ]
                all_images = (all_images + 1) * 0.5
                ## unnormalize
                unnormed = self.ds.unnormalize(all_images)
                observations = unnormed[:,:,:self.ds.obs_dim]
                savepath = str(self.results_folder / f'_sample-reference.png')
                plot_samples(savepath, observations, self.renderer)

            if self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                inds = np.random.randint(0, len(self.ds), size=2)
                ## self.ds[i] : [ data , mask ]
                ## self.ds[i][0] : data : [ 1 x T x (obs_dim + act_dim) ]
                ## self.ds[i][0][b,t,:obs_dim] : data[t=0] : [ obs_dim ]
                conditions_l = [
                    [], ## no conditioning
                    [(0,  self.ds.obs_dim, self.ds[inds[0]][0][0, 0, :self.ds.obs_dim])], ## first state conditioning
                    # [(-1, self.ds.obs_dim, self.ds[inds[1]][0][0,-1])], ## last state conditioning
                ]

                for i, conditions in enumerate(conditions_l):
                    ## [ -1, 1 ]
                    all_images = self.ema_model.conditional_sample(batch_size=8, conditions=conditions)
                    ## [ 0, 1 ]
                    all_images = (all_images + 1) * 0.5
                    ## unnormalize
                    ## [ B x 1 x T x (obs_dim + act_dim) ]
                    unnormed = self.ds.unnormalize(all_images)
                    ## [ B x 1 x T x obs_dim ]
                    observations = unnormed[:,:,:,:self.ds.obs_dim]
                    savepath = str(self.results_folder / f'sample-{milestone}-{i}.png')
                    plot_samples(savepath, observations, self.renderer)


            self.step += 1

        print('training completed')

def to_np(x):
    return x.detach().cpu().numpy()

def plot_samples(savepath, samples_l, renderer):
    '''
        samples : [ B x 1 x H x obs_dim ]
    '''
    render_kwargs = {
        'trackbodyid': 2,
        'distance': 10,
        'lookat': [10, 2, 0.5],
        'elevation': 0
    }
    images = []
    for samples in samples_l:
        ## [ H x obs_dim ]
        samples = samples.squeeze(0)
        img = renderer.composite(None, to_np(samples), dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
        images.append(img)

    images = np.concatenate(images, axis=0)
    imageio.imsave(savepath, images)
    print(f'Saved {len(samples_l)} samples to: {savepath}')
