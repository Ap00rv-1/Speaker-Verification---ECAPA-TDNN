"""
Shared utility functions for the speaker verification pipeline.
"""

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16_000


def load_and_preprocess_audio(audio_array, orig_sr, target_sr=SAMPLE_RATE):
    """
    Convert raw audio array to a mono, resampled torch tensor.

    Args:
        audio_array: numpy array from HuggingFace dataset audio column
        orig_sr:     original sample rate
        target_sr:   desired sample rate (default 16kHz for ECAPA-TDNN)

    Returns:
        waveform: torch.Tensor of shape (1, T)
    """
    waveform = torch.tensor(audio_array, dtype=torch.float32)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)

    return waveform


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def canonical_pair_key(idx_a, idx_b):
    """Return a canonical (sorted) key for deduplicating trial pairs."""
    return (min(idx_a, idx_b), max(idx_a, idx_b))


def set_all_seeds(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
