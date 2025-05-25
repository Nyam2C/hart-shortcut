"""This file contains code for diffusion head and loss.

This file is adopted and modified from https://github.com/LTH14/mar/blob/main/models/diffloss.py
"""
from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.experimental
import matplotlib.pyplot as plt
import os

import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections

from flax.training.dynamic_scale import DynamicScale
from flax.training.common_utils import shard

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.sharding import create_sharding, all_gather
from model import DiT
from helper_inference import do_inference

import math

import torch
import torch.nn as nn
from utils.train_state import TrainStateEma
from hart.modules.diffusion import create_diffusion

model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'use_cosine': 0,
    'warmup': 0,
    'dropout': 0.0,
    'hidden_size': 768, # change this!
    'patch_size': 2, # change this!
    'depth': 3, # change this!
    'num_heads': 2, # change this!
    'mlp_ratio': 4, # change this!
    'class_dropout_prob': 0.1,
    'num_classes': 1,
    'denoise_timesteps': 1,
    'cfg_scale': 4.0,
    'target_update_rate': 0.999,
    'use_ema': 0,
    'use_stable_vae': 1,
    'sharding': 'dp', # dp or fsdp.
    't_sampling': 'discrete-dt',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 8, # Make sure its a divisor of batch size.
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut' # or naive.
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'shortcut',
    'name': 'shortcut_{dataset_name}',
})
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

class DiffLoss(nn.Module):
    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        sampler="iddpm",
    ):
        super().__init__()
        self.in_channels = 4 #수정해야할지도?
        self.micro_batch = 32
        # reproducibility & devices
        np.random.seed(10)
        print("Using devices", jax.local_devices())
        device_count = len(jax.local_devices())
        global_device_count = jax.device_count()
        print("Device count", device_count)
        print("Global device count", global_device_count)
        local_batch = self.micro_batch // (global_device_count // device_count)
        print("Global Batch: ", self.micro_batch)
        print("Node Batch: ", local_batch)
        print("Device Batch:", local_batch // device_count)

        # effective batch-size with gradient accumulation

        local_batch = self.micro_batch // (global_device_count // device_count)
        print("Micro-batch per node:", self.micro_batch)
        print("Per-device batch:", local_batch)

        # mixed-precision settings
        self.dtype = jnp.bfloat16
        self.dynamic_scaler = DynamicScale()

        # build model with remat (activation checkpointing)
        print(target_channels)
        dit_args = {
            'patch_size': model_config['patch_size'],
            'hidden_size': model_config['hidden_size'],
            'depth': model_config['depth'],
            'num_heads': model_config['num_heads'],
            'mlp_ratio': model_config['mlp_ratio'],
            'out_channels': 4, #수정해야할지도?
            'class_dropout_prob': model_config['class_dropout_prob'],
            'num_classes': model_config['num_classes'],
            'dropout': model_config['dropout'],
            'ignore_dt': not (model_config['train_type'] in ('shortcut', 'livereflow')),
            'dtype': self.dtype,
        }
        self.model_def = DiT(**dit_args)

        # learning rate schedule
        if model_config.use_cosine:
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=model_config['lr'],
                warmup_steps=model_config['warmup'],
                decay_steps=int(1e6),
            )
        elif model_config['warmup'] > 0:
            lr_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=model_config['lr'],
                transition_steps=model_config['warmup'],
            )
        else:
            lr_schedule = lambda step: model_config['lr']

        # optimizer
        adam = optax.adamw(
            learning_rate=lr_schedule,
            b1=model_config['beta1'],
            b2=model_config['beta2'],
            weight_decay=model_config['weight_decay'],
        )
        self.tx = optax.chain(adam)

        cpu = jax.devices('cpu')[0]
        gpu = jax.devices('gpu')[0]

        # initialization function
        def init_fn(rng):
            param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
            example = {
                'obs': jnp.zeros((1, 32, 32, z_channels), dtype=self.dtype),
                't': jnp.zeros((1,), dtype=self.dtype),
                'dt': jnp.zeros((1,), dtype=self.dtype),
                'label': jnp.zeros((1,), dtype=jnp.int32),
            }
            model_rngs = {'params': param_key, 'dropout': dropout_key}
            params = self.model_def.init(
                model_rngs,
                example['obs'],
                example['t'],
                example['dt'],
                example['label']
            )['params']
            opt_state = self.tx.init(params)
            opt_state_cpu = jax.device_put(opt_state, cpu)
            return TrainStateEma.create(
                model_def=self.model_def,
                params=params,
                tx=self.tx,
                rng=rng,
                opt_state=opt_state_cpu,
            )

        # shape inference & sharding
        rng = jax.random.PRNGKey(10)
        state_shape = jax.eval_shape(init_fn, rng)
        data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(model_config.sharding, state_shape)
        self.train_state = jax.jit(init_fn, out_shardings=train_state_sharding)(rng)
        self.shared_data = shard_data

        # load checkpoint (without optimizer state)
        cp = Checkpoint('/content/hart/celeba-shortcut2-every4400001')
        ckpt = cp.load_as_dict()['train_state']
        self.train_state = self.train_state.replace(**ckpt)
        self.train_state = jax.jit(lambda x: x, out_shardings=train_state_sharding)(self.train_state)
        print("Loaded model at step", int(self.train_state.step))
        self.train_state = self.train_state.replace(step=0)
        del cp

    def initialize_weights(self):
        self.net.initialize_weights()

    def forward(self, target, z, mask=None):
        t = torch.randint(
            0,
            self.train_diffusion.num_timesteps,
            (target.shape[0],),
            device=target.device,
        )
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(
            self.net, target / self.vae_scale, t, model_kwargs
        )
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.5, sampler=None):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
        with jax.spmd_mode('allow_all'):
            global_device_count = jax.device_count()
            key = jax.random.PRNGKey(42 + jax.process_index())
            labels_uncond = self.shared_data(jnp.ones((32,), dtype=jnp.int32) * model_config['num_classes']) # Null token
            eps = jax.random.normal(key, (32,32,32,4))

            @partial(jax.jit, static_argnums=(5,))
            def call_model(train_state, images, t, dt, labels, use_ema=True):
                call_fn = train_state.call_model
                output = call_fn(images, t, dt, labels, train=False)
                return output
            denoise_timesteps = 1
            cfg_scale = 1.0
            x0 = []
            images_shape = (32,32,32,4)
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, 1)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            x = jax.random.normal(eps_key, images_shape)
            x = self.shared_data(x)
            x0.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            delta_t = 1.0 / denoise_timesteps
            t = 1
            t_vector = jnp.full((images_shape[0], ), t)
            dt_flow = np.log2(denoise_timesteps).astype(jnp.int32)
            dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
            t_vector, dt_base = self.shared_data(t_vector, dt_base)
            v = call_model(self.train_state, x, t_vector, dt_base, labels_uncond)
            eps = self.shared_data(jax.random.normal(jax.random.fold_in(eps_key, 1), images_shape))
            x1pred = x + v * (1-t)
            x = x1pred * (t+delta_t) + eps * (1-t-delta_t)
        x_np = np.array(x)
        x_torch = torch.from_numpy(x_np)
        x_torch = x_torch.permute(0, 3, 1, 2).contiguous()
        device = torch.device("cuda")
        x_torch = x_torch.to(device, dtype=torch.float16)
        x_torch = x_torch.permute(0, 3, 1, 2).contiguous()
        return x_torch


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        for block in self.res_blocks:
            x = block(x, y)
        o = self.final_layer(x, y)
        return o
    from functools import partial
    
    def forward2(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c
        
        dt_flow = int(math.log2(128))
        batch_size = x.size(0)            # x�� (B, C, H, W) ������ �ټ�
        device     = x.device             # GPU/CPU ��ġ ����
        dt_base    = torch.full(
            (batch_size,),
            dt_flow,
            dtype=torch.int32,
            device=device
        )

        
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            call_fn = train_state.call_model_ema
            output = call_fn(images, t, dt, labels, train=False)
            return output
        cp = Checkpoint("/content/hart/checkpoints/celeba-shortcut2-every4400001")
        replace_dict = cp.load_as_dict()['train_state']
        del replace_dict['opt_state'] # Debug
        train_state = train_state.replace(**replace_dict)
        train_state = train_state.replace(step=0)

        v = call_model(train_state, x, y, dt_base, c)
        delta_t = 1.0 / 128
        x1pred = x + v * (1-t)
        x = x1pred * (t+delta_t) * (1-t-delta_t)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = cond_eps * (1 + cfg_scale) - cfg_scale * uncond_eps
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    from functools import partial

    def forward_with_dpmsolver(self, x, timestep, c):
        model_output = self.forward(x, timestep, c)
        return model_output.chunk(2, dim=1)[0]
