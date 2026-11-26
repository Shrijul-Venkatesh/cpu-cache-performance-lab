import numpy as np
from numba import njit


@njit(cache=True)
def kernel_sequential(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total


@njit(cache=True)
def kernel_strided(arr, stride):
    total = 0.0
    for i in range(0, len(arr), stride):
        total += arr[i]
    return total


@njit(cache=True)
def kernel_gather(arr, indices):
    total = 0.0
    for i in range(len(indices)):
        total += arr[indices[i]]
    return total


@njit(cache=True)
def kernel_pointer_chase(arr, indices):
    """Each step depends on reads from both the index and data arrays."""
    idx = 0
    total = 0.0
    for _ in range(len(indices)):
        idx = indices[idx]
        total += arr[idx]
    return total


def warmup_kernels() -> None:
    sample = np.ones(1024, dtype=np.float64)
    indices = np.arange(1024, dtype=np.int64)
    kernel_sequential(sample)
    kernel_strided(sample, 1)
    kernel_gather(sample, indices)
    kernel_pointer_chase(sample, indices)


def build_chase_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return one random cycle that visits every index once."""
    permutation = rng.permutation(n)
    indices = np.empty(n, dtype=np.int64)
    indices[permutation[:-1]] = permutation[1:]
    indices[permutation[-1]] = permutation[0]
    return indices
