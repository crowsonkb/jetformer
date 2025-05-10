#!/usr/bin/env python3

"""Implementation of Jetformer."""

import argparse
from copy import deepcopy
from functools import lru_cache, wraps
import math
from pathlib import Path

from einops import rearrange
from PIL import Image
import scipy.stats
import torch
from torch import distributed as dist, distributions as D, nn, optim
from torch.distributed import nn as dnn
from torch.nn import functional as F
from torch.utils import data
import torch_dist_utils as du
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import trange, tqdm

print = tqdm.external_write_mode()(print)
print0 = tqdm.external_write_mode()(du.print0)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class FolderOfImages(data.Dataset):
    """Recursively finds all images in a directory. It does not support
    classes/targets."""

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = nn.Identity() if transform is None else transform
        self.paths = sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in self.IMG_EXTENSIONS
        )

    def __repr__(self):
        return f'FolderOfImages(root="{self.root}", len: {len(self)})'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        path = self.paths[key]
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(0, dtype=torch.long)


def zero_init(module):
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


def checkpoint(func, use_reentrant=False):
    @wraps(func)
    def inner(*args, **kwargs):
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                func, *args, **kwargs, use_reentrant=use_reentrant
            )
        else:
            return func(*args, **kwargs)

    return inner


@torch.compile(dynamic=True)
def gmm_loglik(
    x: torch.Tensor, logits: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Diagonal Gaussian Mixture Model log likelihood.

    Args:
        x (torch.Tensor): The input tensor, shape (..., dim).
        logits (torch.Tensor): The logits tensor, shape (..., num_components).
        mean (torch.Tensor): The mean tensor, shape (..., num_components, dim).
        logvar (torch.Tensor): The log variance tensor, shape (..., num_components, dim).

    Returns:
        torch.Tensor: The log likelihood tensor, shape (...).
    """
    tmp1 = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    tmp2 = -0.5 * torch.sum(((x.unsqueeze(-2) - mean) ** 2 * torch.exp(-logvar) + logvar), dim=-1)
    tmp3 = -0.5 * x.shape[-1] * math.log(2 * math.pi)
    return torch.logsumexp(tmp1 + tmp2, dim=-1) + tmp3


def gmm_sample(
    logits: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    temperature_weights: float = 1.0,
    top_p_weights: float = 1.0,
    temperature_component: float = 1.0,
    top_p_component: float = 1.0,
) -> torch.Tensor:
    """Samples from a diagonal Gaussian Mixture Model.

    Args:
        logits (torch.Tensor): The logits tensor, shape (..., num_components).
        mean (torch.Tensor): The mean tensor, shape (..., num_components, dim).
        logvar (torch.Tensor): The log variance tensor, shape (..., num_components, dim).
        temperature_weights (float): The temperature for the weights.
        top_p_weights (float): The top-p threshold for the weights.
        temperature_component (float): The temperature for the component.
        top_p_component (float): The top-p threshold for the component.

    Returns:
        torch.Tensor: The sample tensor, shape (..., dim).
    """
    # apply temperature and top-p to the weights
    logits = logits / temperature_weights
    if top_p_weights < 1.0:
        probs = torch.softmax(logits, dim=-1)
        probs_sorted, indices = torch.sort(probs, dim=-1, descending=True, stable=True)
        probs_cumsum = torch.cumsum(probs_sorted, dim=-1)
        drop = probs_cumsum[..., :-1] >= top_p_weights
        drop = torch.cat((drop.new_zeros(*drop.shape[:-1], 1), drop), dim=-1)
        drop_unsorted = torch.empty_like(drop).scatter_(-1, indices, drop)
        logits = torch.masked_fill(logits, drop_unsorted, float("-inf"))

    # sample the component and gather its mean and logvar
    gumbel = torch.empty_like(logits).exponential_().log_().neg_()
    index = torch.argmax(logits + gumbel, dim=-1, keepdim=True)
    index = index.expand(*index.shape[:-1], mean.shape[-1])
    mean = mean.gather(-2, index.unsqueeze(-2)).squeeze(-2)
    logvar = logvar.gather(-2, index.unsqueeze(-2)).squeeze(-2)

    # sample the normal noise
    r2 = scipy.stats.chi2.ppf(top_p_component, mean.shape[-1]).item()
    noise = torch.randn_like(mean)
    while True:
        cond = torch.sum(noise**2, dim=-1, keepdim=True) <= r2
        if torch.all(cond):
            break
        noise = torch.where(cond, noise, torch.randn_like(noise))
    noise = noise * temperature_component**0.5

    # return the final sample
    return mean + noise * torch.exp(logvar / 2)


class ChromaSubsample(nn.Module):
    """Converts RGB images to YUV and 2x subsamples the chroma channels."""

    def __init__(self):
        super().__init__()
        filt_to_yuv = torch.tensor(
            [[0.2126, 0.7152, 0.0722], [-0.09991, -0.33609, 0.436], [0.615, -0.55861, -0.05639]]
        )
        filt_to_rgb = torch.linalg.inv(filt_to_yuv)
        self.register_buffer("filt_to_yuv", filt_to_yuv[..., None, None])
        self.register_buffer("filt_to_rgb", filt_to_rgb[..., None, None])

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Input must have shape (N, 3, H, W).")
        # convert to YUV
        yuv = F.conv2d(rgb, self.filt_to_yuv)
        y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
        # subsample chroma
        u = F.avg_pool2d(u, 2)
        v = F.avg_pool2d(v, 2)
        # pack
        y = F.pixel_unshuffle(y, 2)
        return torch.cat((y, u, v), dim=1)

    def inverse(self, yuv):
        if yuv.ndim != 4 or yuv.shape[1] != 6:
            raise ValueError("Input must have shape (N, 6, H / 2, W / 2).")
        # unpack
        y, u, v = yuv[:, 0:4], yuv[:, 4:5], yuv[:, 5:6]
        y = F.pixel_shuffle(y, 2)
        # interpolate chroma
        u = F.interpolate(u, scale_factor=2, mode="bilinear", align_corners=False)
        v = F.interpolate(v, scale_factor=2, mode="bilinear", align_corners=False)
        # convert to RGB
        yuv = torch.cat((y, u, v), dim=1)
        return F.conv2d(yuv, self.filt_to_rgb)


class RoPE(nn.Module):
    def __init__(self, rot_dim: int, pos_dim: int = 1, theta: float = 10000.0):
        super().__init__()
        self.rot_dim = rot_dim
        self.pos_dim = pos_dim
        self.theta = theta
        freqs = theta ** torch.linspace(0, -1, rot_dim // (pos_dim * 2) + 1)[:-1]
        self.register_buffer("freqs", freqs)

    def extra_repr(self):
        return f"rot_dim={self.rot_dim}, pos_dim={self.pos_dim}, theta={self.theta}"

    @staticmethod
    def make_pos(*shape, device=None):
        ranges = [torch.arange(s, device=device) for s in shape]
        pos = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)
        return pos.flatten(0, len(shape) - 1)

    @staticmethod
    @lru_cache(maxsize=1)
    def make_cos_sin(pos, freqs):
        tmp = (pos.unsqueeze(-1) * freqs).flatten(-2).unsqueeze(1)
        return torch.cos(tmp), torch.sin(tmp)

    @staticmethod
    def apply_rotary_emb(x, cos, sin):
        dim = cos.shape[-1]
        x1, x2, x3 = x[..., :dim], x[..., dim : dim * 2], x[..., dim * 2 :]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1.to(x3.dtype), y2.to(x3.dtype), x3), dim=-1)

    def forward(self, pos, q, k):
        # pos is shape (batch, seq, pos_dim)
        # q and k are shape (batch, num_heads, seq, dim)
        cos, sin = self.make_cos_sin(pos, self.freqs)
        q = self.apply_rotary_emb(q, cos, sin)
        k = self.apply_rotary_emb(k, cos, sin)
        return q, k


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.linear_1 = nn.Linear(dim, dim * 4)
        self.linear_2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        return x + skip


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head_dim = 64
        self.num_heads = dim // self.head_dim
        self.norm = nn.LayerNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.pos_emb = RoPE(self.head_dim // 2)

    def init_cache(self, batch_size, seq_len, device=None, dtype=None):
        k = torch.zeros(
            batch_size, self.num_heads, seq_len, self.head_dim, device=device, dtype=dtype
        )
        v = torch.zeros(
            batch_size, self.num_heads, seq_len, self.head_dim, device=device, dtype=dtype
        )
        return k, v

    def forward(self, x, cache=None, index=None):
        skip = x
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n s (t h d) -> t n h s d", t=3, d=self.head_dim)
        if cache is None:
            pos = torch.arange(k.shape[2], device=k.device)[None, :, None]
            q, k = self.pos_emb(pos, q, k)
            x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            end_index = index + 1
            pos = torch.arange(index, end_index, device=k.device)[None, :, None]
            q, k = self.pos_emb(pos, q, k)
            cache[0][:, :, index:end_index] = k
            cache[1][:, :, index:end_index] = v
            k_in = cache[0][:, :, :end_index]
            v_in = cache[1][:, :, :end_index]
            x = F.scaled_dot_product_attention(q, k_in, v_in)
        x = rearrange(x, "n h s d -> n s (h d)")
        x = self.out_proj(x)
        return x + skip


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = SelfAttention(dim)
        self.ffn = FFN(dim)

    def init_cache(self, batch_size, seq_len, device=None, dtype=None):
        return self.attn.init_cache(batch_size, seq_len, device, dtype)

    def forward(self, x, cache=None, index=None):
        x = self.attn(x, cache, index)
        x = self.ffn(x)
        return x


class GMMHead(nn.Module):
    def __init__(self, in_features, out_features, num_components):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_components = num_components
        self.logits_proj = nn.Linear(in_features, num_components)
        self.mean_proj = nn.Linear(in_features, num_components * out_features)
        self.logvar_proj = zero_init(nn.Linear(in_features, num_components * out_features))

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, num_components={self.num_components}"

    def forward(self, x):
        logits = self.logits_proj(x)
        mean = self.mean_proj(x).unflatten(-1, (self.num_components, self.out_features))
        logvar = self.logvar_proj(x).unflatten(-1, (self.num_components, self.out_features))
        return logits, mean, logvar


class GIVT(nn.Module):
    def __init__(self, in_dim, dim, depth, num_components, num_classes):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        self.cls_emb = nn.Embedding(num_classes, dim)
        self.blocks = nn.ModuleList(TransformerBlock(dim) for _ in range(depth))
        self.out_norm = nn.LayerNorm((dim,))
        self.out_head = GMMHead(dim, in_dim, num_components)

    def init_cache(self, batch_size, seq_len, device=None, dtype=None):
        return [block.init_cache(batch_size, seq_len, device, dtype) for block in self.blocks]

    def forward(self, x, y, cache=None, index=None):
        x = self.in_proj(x)
        y = self.cls_emb(y)
        x = torch.cat((y[:, None], x), dim=1)
        x = x if cache is None else x[:, -1:].contiguous()
        cache = [None] * len(self.blocks) if cache is None else cache
        for block, cache_block in zip(self.blocks, cache):
            x = checkpoint(block)(x, cache_block, index)
        x = self.out_norm(x)
        x = self.out_head(x)
        return x


class SelfAttentionForJet(nn.Module):
    def __init__(self, dim, kernel_size=(7, 7)):
        super().__init__()
        self.head_dim = 64
        self.num_heads = dim // self.head_dim
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.pos_emb = RoPE(self.head_dim // 2, pos_dim=2)

    @staticmethod
    @lru_cache(maxsize=1)
    def make_attn_mask(pos, padding_mask, kernel_size):
        # pos is (n, s, 2), dtype long
        # padding mask is (n, s), dtype bool
        # return type is (n, 1, s, s), dtype bool
        mh1 = pos[:, :, None, 0] <= pos[:, None, :, 0] + kernel_size[0] // 2
        mh2 = pos[:, :, None, 0] > pos[:, None, :, 0] - (kernel_size[0] + 1) // 2
        mw1 = pos[:, :, None, 1] <= pos[:, None, :, 1] + kernel_size[1] // 2
        mw2 = pos[:, :, None, 1] > pos[:, None, :, 1] - (kernel_size[1] + 1) // 2
        mask1 = mh1 & mh2 & mw1 & mw2
        mask2 = padding_mask[:, None, :] & padding_mask[:, :, None]
        return (mask1 & mask2)[:, None, :, :]

    def forward(self, x, pos, padding_mask):
        skip = x
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n s (t h d) -> t n h s d", t=3, d=self.head_dim)
        q, k = self.pos_emb(pos, q, k)
        attn_mask = self.make_attn_mask(pos, padding_mask, self.kernel_size)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask)
        x = rearrange(x, "n h s d -> n s (h d)")
        x = self.out_proj(x)
        return x + skip


class TransformerBlockForJet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = SelfAttentionForJet(dim)
        self.ffn = FFN(dim)

    def forward(self, x, pos, padding_mask):
        x = self.attn(x, pos, padding_mask)
        x = self.ffn(x)
        return x


class TransformerForJet(nn.Module):
    def __init__(self, in_dim, dim, depth, out_scale):
        super().__init__()
        self.out_scale = out_scale
        self.in_proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([TransformerBlockForJet(dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = zero_init(nn.Linear(dim, in_dim * 2))

    def forward(self, x, pos, padding_mask):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, pos, padding_mask)
        x = self.out_norm(x)
        x = self.out_proj(x)
        a, b = x.chunk(2, dim=-1)
        return a * self.out_scale, b * self.out_scale


class AffineCoupling(nn.Module):
    def __init__(self, in_dim, dim, depth, out_scale):
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError("in_dim must be even")
        self.transformer_1 = TransformerForJet(in_dim // 2, dim, depth, out_scale)
        self.transformer_2 = TransformerForJet(in_dim // 2, dim, depth, out_scale)
        index_1, index_2 = torch.randperm(in_dim).chunk(2)
        self.register_buffer("index_1", index_1.contiguous())
        self.register_buffer("index_2", index_2.contiguous())

    def forward(self, x, pos, padding_mask):
        x1, x2 = x[..., self.index_1], x[..., self.index_2]
        a1, b1 = checkpoint(self.transformer_1)(x1, pos, padding_mask)
        x2 = x2 * torch.exp(b1) + a1
        logdet_1 = torch.sum(b1 * padding_mask.unsqueeze(-1), dim=(-2, -1))
        a2, b2 = checkpoint(self.transformer_2)(x2, pos, padding_mask)
        x1 = x1 * torch.exp(b2) + a2
        logdet_2 = torch.sum(b2 * padding_mask.unsqueeze(-1), dim=(-2, -1))
        x = torch.empty_like(x)
        x[..., self.index_1] = x1.to(x.dtype)
        x[..., self.index_2] = x2.to(x.dtype)
        return x, logdet_1 + logdet_2

    def inverse(self, x, pos, padding_mask):
        x1, x2 = x[..., self.index_1], x[..., self.index_2]
        a2, b2 = self.transformer_2(x2, pos, padding_mask)
        x1 = (x1 - a2) * torch.exp(-b2)
        a1, b1 = self.transformer_1(x1, pos, padding_mask)
        x2 = (x2 - a1) * torch.exp(-b1)
        x = torch.empty_like(x)
        x[..., self.index_1] = x1.to(x.dtype)
        x[..., self.index_2] = x2.to(x.dtype)
        return x


class Jet(nn.Module):
    def __init__(self, in_dim, dim, inner_depth, depth):
        super().__init__()
        if depth % 2 != 0:
            raise ValueError("depth must be even")
        self.blocks = nn.ModuleList(
            [AffineCoupling(in_dim, dim, inner_depth, 2 / depth) for _ in range(depth // 2)]
        )

    def forward(self, x, pos, padding_mask):
        logdets = []
        for block in self.blocks:
            x, logdet = block(x, pos, padding_mask)
            logdets.append(logdet)
        return x, torch.stack(logdets).sum(dim=0)

    def inverse(self, x, pos, padding_mask):
        for block in reversed(self.blocks):
            x = block.inverse(x, pos, padding_mask)
        return x


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True, help="path to dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size per device")
    parser.add_argument("--uncond", action="store_true", help="dataset is unconditional")
    parser.add_argument("--sigma", type=float, default=0.05, help="noise augmentation std")
    parser.add_argument("--cond-scale", type=float, default=1.0, help="condition scale")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p threshold")
    args = parser.parse_args()

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cs = ChromaSubsample().to(device)
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    if not args.uncond:
        dataset = datasets.ImageFolder(args.dataset, transform=tf)
    else:
        dataset = FolderOfImages(args.dataset, transform=tf)
    num_classes = 0 if args.uncond else len(dataset.classes)
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=16,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # TODO: make this configurable
    patch_dim = 96  # 8x8 patches, chroma subsampled
    latent_dim = 24  # number of dimensions per patch modeled by the GIVT
    prior_dim = patch_dim - latent_dim
    num_components = 1024  # number of components in the GIVT's Gaussian mixture model output
    spatial = 32, 32  # height and width of the images, in patches
    seq_len = spatial[0] * spatial[1]
    ema_decay = 0.99

    # TODO: make the model configurable and make the default larger, this is just for testing
    givt_raw = GIVT(
        in_dim=latent_dim,
        dim=768,
        depth=12,
        num_components=num_components,
        num_classes=num_classes + 1,  # add one for the unconditional class
    ).to(device)

    jet_raw = Jet(
        in_dim=patch_dim,
        dim=512,
        inner_depth=3,
        depth=12,
    ).to(device)

    du.broadcast_tensors(givt_raw.parameters())
    du.broadcast_tensors(jet_raw.parameters())
    du.broadcast_tensors(jet_raw.buffers())

    num_params_givt = sum(p.numel() for p in givt_raw.parameters())
    num_params_jet = sum(p.numel() for p in jet_raw.parameters())
    print0(f"GIVT parameters: {num_params_givt:,}")
    print0(f"Jet parameters: {num_params_jet:,}")
    print0(f"Total parameters: {num_params_givt + num_params_jet:,}")

    givt_ema = deepcopy(givt_raw).eval().requires_grad_(False)
    jet_ema = deepcopy(jet_raw).eval().requires_grad_(False)

    givt = nn.parallel.DistributedDataParallel(givt_raw, device_ids=[device], output_device=device)
    jet = nn.parallel.DistributedDataParallel(jet_raw, device_ids=[device], output_device=device)

    # TODO: don't weight decay layernorms etc
    opt = optim.AdamW(
        [
            {"params": givt.parameters()},
            {"params": jet.parameters()},
        ],
        lr=5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    sched = optim.lr_scheduler.LambdaLR(opt, lambda step: 1 - 0.9 ** (step + 1))

    pos = RoPE.make_pos(*spatial, device=device).unsqueeze(0)
    padding_mask = torch.ones(1, seq_len, device=device, dtype=torch.bool)

    epoch = 0
    step = 0

    # not correct classifier-free guidance, but correct cfg can break
    def apply_cond_scale(logits, mean, logvar, cond_scale):
        lc, lu = logits.chunk(2)
        mc, mu = mean.chunk(2)
        lvc, lvu = logvar.chunk(2)
        logits = torch.lerp(lu, lc, cond_scale)
        mean = torch.lerp(mu, mc, cond_scale)
        logvar = torch.lerp(lvu, lvc, cond_scale)
        return logits, mean, logvar

    @torch.no_grad()
    def demo():
        demo_bs = 36
        demo_bs_per_rank = math.ceil(demo_bs / world_size)

        x_latent = torch.empty(demo_bs_per_rank, seq_len, latent_dim, device=device)

        if num_classes > 0:
            y = torch.randint(num_classes, (demo_bs_per_rank,), device=device)
        else:
            y = torch.zeros(demo_bs_per_rank, device=device, dtype=torch.long)
        y_in = torch.cat((y, torch.full((demo_bs_per_rank,), num_classes, device=device)))

        cache = givt_ema.init_cache(
            demo_bs_per_rank * 2, seq_len, device=device, dtype=torch.bfloat16
        )

        # sample the latent from the GIVT
        for index in trange(seq_len, disable=rank != 0):
            x_latent_in = torch.cat((x_latent[:, :index], x_latent[:, :index]))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gmm = givt_ema(x_latent_in, y_in, cache, index)
                gmm = apply_cond_scale(*gmm, args.cond_scale)
                sample = gmm_sample(*gmm, top_p_weights=args.top_p, top_p_component=args.top_p)
                x_latent[:, index] = sample[:, 0]

        # decode the latent
        x_prior = torch.randn(demo_bs_per_rank, seq_len, prior_dim, device=device)
        x_flowed = torch.cat((x_latent, x_prior), dim=-1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x_noised = jet_ema.inverse(x_flowed, pos, padding_mask)

        # denoise
        with torch.enable_grad():
            x_noised_ = x_noised.clone().requires_grad_()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x_flowed, logdet = jet_ema(x_noised_, pos, padding_mask)
                x_latent, x_prior = x_flowed[..., :latent_dim], x_flowed[..., latent_dim:]
                x_latent_in = torch.cat((x_latent[:, :-1], x_latent[:, :-1]))
                gmm = givt_ema(x_latent_in, y_in)
                gmm = apply_cond_scale(*gmm, args.cond_scale)
                ll_gmm = torch.sum(gmm_loglik(x_latent, *gmm) * padding_mask)
                ll_prior = torch.sum(D.Normal(0, 1).log_prob(x_prior) * padding_mask.unsqueeze(-1))
                ll = ll_gmm + ll_prior + torch.sum(logdet)
            grad = torch.autograd.grad(ll, x_noised_)[0]
        x = x_noised + args.sigma**2 * grad

        # unpatch and unsubsample chroma
        x = rearrange(
            x, "n (h w) (hh ww c) -> n c (h hh) (w ww)", h=spatial[0], w=spatial[1], hh=4, ww=4
        )
        x = cs.inverse(x)
        x = x * 0.5 + 0.5
        x = torch.clamp(x, 0, 1)

        # make the grid and save it
        x = torch.cat(dnn.all_gather(x))[:demo_bs]
        grid = rearrange(x, "(hh ww) c h w -> c (hh h) (ww w)", hh=6, ww=6)
        if rank == 0:
            TF.to_pil_image(grid.float().cpu()).save("demo.png")

    # TODO: save the model

    while True:
        sampler.set_epoch(epoch)

        for x, y in tqdm(dataloader, smoothing=0.1, disable=rank != 0):
            x = x.to(device)
            y = y.to(device)
            x = cs(x)

            # make demo grid
            if step % 500 == 0:
                print0("Sampling demo grid...")
                demo()

            # patch 4x4 (since chroma subsampling already 2x2 patched the luma channel)
            x = rearrange(x, "n c (h hh) (w ww) -> n (h w) (hh ww c)", hh=4, ww=4)

            # augment the images with noise
            x_noised_dist = D.Normal(x, args.sigma)
            x_noised = x_noised_dist.sample()

            # 10% condition dropout
            y = torch.where(torch.rand_like(y, dtype=torch.float32) < 0.1, num_classes, y)

            # forward pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # encode the noised images to latents
                x_flowed, logdet = jet(x_noised, pos, padding_mask)
                x_latent, x_prior = x_flowed[..., :latent_dim], x_flowed[..., latent_dim:]

                # predict the next latent patch
                gmm = givt(x_latent[:, :-1], y)

                # compute the loss (nats per dimension)
                ll_p = torch.sum(x_noised_dist.log_prob(x_noised) * padding_mask.unsqueeze(-1))
                ll_gmm = torch.sum(gmm_loglik(x_latent, *gmm) * padding_mask)
                ll_prior = torch.sum(D.Normal(0, 1).log_prob(x_prior) * padding_mask.unsqueeze(-1))
                ll_q = ll_gmm + ll_prior + torch.sum(logdet)
                loss = (ll_p - ll_q) / x_noised.numel()

            # backward pass and optimizer step
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            # TODO: use a k-diffusion style EMA warmup
            ema_update(givt_raw, givt_ema, ema_decay)
            ema_update(jet_raw, jet_ema, ema_decay)

            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            if step % 1 == 0:
                print0(f"step: {step}, loss: {loss:g}")
            step += 1

        epoch += 1


if __name__ == "__main__":
    main()
