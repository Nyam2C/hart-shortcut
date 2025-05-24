import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
from functools import partial
from absl import app, flags

flags.DEFINE_integer('inference_timesteps', 128, 'Number of timesteps for inference.')
flags.DEFINE_integer('inference_generations', 4096, 'Number of generations for inference.')
flags.DEFINE_float('inference_cfg_scale', 1.0, 'CFG scale for inference.')

def do_inference(
    FLAGS,
    train_state,
    shard_data,
):
    with jax.spmd_mode('allow_all'):
        global_device_count = jax.device_count()
        key = jax.random.PRNGKey(42 + jax.process_index())
        labels_uncond = shard_data(jnp.ones((32,), dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token
        eps = jax.random.normal(key, (32,32,32,4))
        
        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False)
            return output
        denoise_timesteps = 1
        num_generations = FLAGS.inference_generations
        cfg_scale = FLAGS.inference_cfg_scale
        x0 = []
        images_shape = (32,32,32,4)
        key = jax.random.PRNGKey(42)
        key = jax.random.fold_in(key, 1)
        key = jax.random.fold_in(key, jax.process_index())
        eps_key, label_key = jax.random.split(key)
        x = jax.random.normal(eps_key, images_shape)
        x = shard_data(x)
        x0.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
        delta_t = 1.0 / denoise_timesteps
        for ti in range(denoise_timesteps):
          t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
          t_vector = jnp.full((images_shape[0], ), t)
          dt_flow = np.log2(denoise_timesteps).astype(jnp.int32)
          dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
          t_vector, dt_base = shard_data(t_vector, dt_base)
          v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
          eps = shard_data(jax.random.normal(jax.random.fold_in(eps_key, ti), images_shape))
          x1pred = x + v * (1-t)
          x = x1pred * (t+delta_t) + eps * (1-t-delta_t)