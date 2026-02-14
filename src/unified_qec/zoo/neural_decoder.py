"""
Neural Syndrome Decoder

A real JAX/Flax neural network for syndrome decoding. Uses a
Transformer-style architecture with residual connections, layer norm,
and multi-head self-attention to decode syndrome vectors into
observable flip predictions.

This is the decoder that gets checkpointed into the Model Zoo and
loaded by the Sinter neural backend.

Architecture:
    Input (num_detectors) → Embedding → N × TransformerBlock → Dense → Output (num_observables)

Each TransformerBlock:
    LayerNorm → MultiHeadSelfAttention → Residual → LayerNorm → FFN → Residual
"""

import numpy as np
from typing import Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    random = None

try:
    import flax.linen as nn
    from flax.training import train_state
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = None
    train_state = None
    optax = None


def _check_deps():
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for the neural decoder. "
            "Install with: pip install unified-qec[jax]"
        )
    if not FLAX_AVAILABLE:
        raise ImportError(
            "Flax and Optax are required for the neural decoder. "
            "Install with: pip install flax optax"
        )


class SyndromeEmbedding(nn.Module):
    """Project binary syndrome vector into dense embedding space."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, name="proj")(x)
        x = nn.gelu(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and FFN.

    Parameters
    ----------
    hidden_dim : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    ffn_dim : int
        Feed-forward expansion dimension.
    dropout_rate : float
        Dropout probability (training only).
    """
    hidden_dim: int
    num_heads: int
    ffn_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Self-attention with residual
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        # Reshape for attention: (batch, seq_len=1, hidden_dim) → attend over features
        batch_size = x.shape[0]
        seq_len = self.hidden_dim // self.num_heads
        x_attn = x.reshape(batch_size, seq_len, self.num_heads)
        x_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            name="attn",
        )(x_attn, x_attn, deterministic=deterministic)
        x_attn = x_attn.reshape(batch_size, -1)
        # Project back to hidden_dim if shape changed
        if x_attn.shape[-1] != self.hidden_dim:
            x_attn = nn.Dense(self.hidden_dim, name="attn_proj")(x_attn)
        x = residual + x_attn
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # FFN with residual
        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = nn.Dense(self.ffn_dim, name="ffn1")(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim, name="ffn2")(x)
        x = residual + x
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        return x


class NeuralSyndromeDecoder(nn.Module):
    """Neural network decoder for quantum error correction syndromes.

    Transformer-based architecture that learns to map syndrome vectors
    to observable flip predictions. Designed for integration with the
    Sinter pipeline via the Model Zoo.

    Parameters
    ----------
    num_detectors : int
        Number of syndrome bits (input dimension).
    num_observables : int
        Number of logical observable predictions (output dimension).
    hidden_dim : int
        Model internal dimension.
    num_layers : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads.
    ffn_dim : int
        Feed-forward expansion dimension. Default: 4 × hidden_dim.
    dropout_rate : float
        Dropout rate during training.
    """
    num_detectors: int
    num_observables: int
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 0  # 0 means 4 * hidden_dim
    dropout_rate: float = 0.1

    def setup(self):
        _check_deps()
        ffn = self.ffn_dim if self.ffn_dim > 0 else 4 * self.hidden_dim
        self.embed = SyndromeEmbedding(hidden_dim=self.hidden_dim)
        self.blocks = [
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ffn_dim=ffn,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = nn.LayerNorm(name="final_norm")
        self.head = nn.Dense(self.num_observables, name="head")

    def __call__(self, x, deterministic: bool = True):
        """Forward pass.

        Parameters
        ----------
        x : jnp.ndarray, shape (batch, num_detectors)
            Binary syndrome vectors (float32).
        deterministic : bool
            If True, disable dropout.

        Returns
        -------
        jnp.ndarray, shape (batch, num_observables)
            Logits for observable flip predictions.
        """
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        x = self.final_norm(x)
        x = self.head(x)
        return x

    def predict(self, params, syndromes: np.ndarray) -> np.ndarray:
        """Predict observable flips from syndrome vectors.

        Parameters
        ----------
        params : dict
            Model parameters (from checkpoint).
        syndromes : ndarray, shape (batch, num_detectors)
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape (batch, num_observables)
            Predicted observable flips (binary).
        """
        _check_deps()
        logits = jax.jit(self.apply)(params, jnp.array(syndromes, dtype=jnp.float32))
        return np.array(logits > 0.0, dtype=np.uint8)


def create_train_state(
    rng: "jax.random.PRNGKey",
    num_detectors: int,
    num_observables: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
) -> "train_state.TrainState":
    """Create an initialized training state for the neural decoder.

    Parameters
    ----------
    rng : PRNGKey
        Random number generator key.
    num_detectors : int
        Number of syndrome bits.
    num_observables : int
        Number of observable predictions.
    hidden_dim : int
        Model dimension.
    num_layers : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads.
    learning_rate : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay coefficient.

    Returns
    -------
    TrainState
        Flax training state with model params, optimizer, and apply_fn.
    """
    _check_deps()

    model = NeuralSyndromeDecoder(
        num_detectors=num_detectors,
        num_observables=num_observables,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    dummy_input = jnp.zeros((1, num_detectors), dtype=jnp.float32)
    params = model.init(rng, dummy_input)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def train_step(state, syndromes, labels):
    """Single training step for the neural decoder.

    Parameters
    ----------
    state : TrainState
        Current training state.
    syndromes : jnp.ndarray, shape (batch, num_detectors)
        Input syndrome vectors.
    labels : jnp.ndarray, shape (batch, num_observables)
        Target observable flips.

    Returns
    -------
    state : TrainState
        Updated training state.
    loss : float
        Binary cross-entropy loss.
    """
    _check_deps()

    def loss_fn(params):
        logits = state.apply_fn(params, syndromes, deterministic=False)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def generate_training_data(
    circuit,
    num_shots: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data from a Stim circuit.

    Parameters
    ----------
    circuit : stim.Circuit
        Stim circuit with noise.
    num_shots : int
        Number of samples to generate.

    Returns
    -------
    syndromes : ndarray, shape (num_shots, num_detectors)
        Detection event vectors.
    observables : ndarray, shape (num_shots, num_observables)
        Observable flip labels.
    """

    sampler = circuit.compile_detector_sampler()
    det_data, obs_data = sampler.sample(
        num_shots, separate_observables=True
    )
    return det_data.astype(np.float32), obs_data.astype(np.float32)
