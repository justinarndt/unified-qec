"""
Model Zoo: Versioned model artifacts for QEC decoders.

Provides Orbax-based checkpointing, HuggingFace Hub distribution,
and a real neural decoder architecture for syndrome decoding.
"""

from unified_qec.zoo.checkpoint import ModelCheckpoint as ModelCheckpoint
from unified_qec.zoo.checkpoint import NumpyCheckpoint as NumpyCheckpoint
from unified_qec.zoo.hub import ZooManager as ZooManager

# Neural decoder requires JAX + Flax — import is conditional
try:
    from unified_qec.zoo.neural_decoder import (
        NeuralSyndromeDecoder as NeuralSyndromeDecoder,
    )
except (ImportError, AttributeError):
    pass
