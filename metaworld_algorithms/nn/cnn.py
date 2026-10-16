"""Mnih et al. (2015) DQN CNN encoder implemented in Flax."""
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


class MnihCNN(nn.Module):
    """CNN from Mnih et al. (2015) Nature DQN.

    Architecture:
        Conv(32, 8x8, stride 4) -> ReLU
        Conv(64, 4x4, stride 2) -> ReLU
        Conv(64, 3x3, stride 1) -> ReLU
        Flatten -> Linear(512) -> ReLU

    Input: (H, W, C) image, values in [0, 255] uint8 or [0, 1] float.
    Output: (512,) feature vector.
    """

    output_dim: int = 512
    activation_fn: Callable = jax.nn.relu

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Normalise uint8 to [0, 1]
        if x.dtype == jnp.uint8:
            x = x.astype(jnp.float32) / 255.0
        else:
            x = x.astype(jnp.float32)

        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = self.activation_fn(x)
        x = x.reshape(*x.shape[:-3], -1)  # flatten spatial dims, keep batch dims
        x = nn.Dense(self.output_dim)(x)
        x = self.activation_fn(x)
        return x


class DualCameraCNN(nn.Module):
    """Processes two (H, W, C) RGB frames with a shared MnihCNN and concatenates outputs.

    Input: dict with keys 'base' and 'hand', each (H, W, 3).
    Output: (2 * output_dim,) feature vector.
    """

    output_dim: int = 512

    @nn.compact
    def __call__(self, base_rgb: jax.Array, hand_rgb: jax.Array) -> jax.Array:
        cnn = MnihCNN(output_dim=self.output_dim)
        base_feat = cnn(base_rgb)
        hand_feat = cnn(hand_rgb)
        return jnp.concatenate([base_feat, hand_feat], axis=-1)
