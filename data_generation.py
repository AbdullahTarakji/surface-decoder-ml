"""
data_generation.py
==================

Single source of truth for syndrome data in the surface-decoder-ml project.

Everything downstream (EDA, MWPM baseline, MLP training, CNN training) imports
from this module. Do not generate Stim data anywhere else.

Usage
-----
    from data_generation import generate_dataset

    X, y, dem = generate_dataset(distance=3, rounds=3, noise_p=1e-3,
                                 n_shots=1_000_000, seed=42)

    # X: (n_shots, n_detectors) bool array — the syndromes
    # y: (n_shots,) bool array            — the logical-flip labels
    # dem: stim.DetectorErrorModel        — for MWPM baseline via PyMatching
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import stim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder where generated datasets are cached. Created on first use.
# Datasets are NOT committed to Git (see .gitignore) because they are large
# and trivially regenerable from (distance, rounds, noise_p, n_shots, seed).
CACHE_DIR = Path(__file__).parent / "data_cache"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    distance: int,
    rounds: int,
    noise_p: float,
    n_shots: int,
    seed: int = 42,
    code_task: str = "surface_code:rotated_memory_z",
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, stim.DetectorErrorModel]:
    """
    Generate (or load from cache) a labeled syndrome dataset for the rotated
    surface code under circuit-level depolarizing noise.

    Parameters
    ----------
    distance : int
        Code distance d. Must be odd. Typical values: 3, 5, 7.
    rounds : int
        Number of stabilizer measurement rounds T. Should be >= distance for
        meaningful threshold behavior under circuit-level noise.
    noise_p : float
        Physical error rate. Applied uniformly to all noise channels
        (after_clifford_depolarization, after_reset_flip_probability,
        before_measure_flip_probability, before_round_data_depolarization).
    n_shots : int
        Number of independent (syndrome, logical_flip) pairs to sample.
    seed : int, default 42
        Random seed for Stim's sampler. Same seed + same parameters →
        bit-identical output.
    code_task : str, default "surface_code:rotated_memory_z"
        Stim code task string. Other options:
        - "surface_code:rotated_memory_x"
        - "surface_code:unrotated_memory_z"
        - "surface_code:unrotated_memory_x"
    use_cache : bool, default True
        If True, look up a cached .npz first; if found, skip Stim and load
        from disk. If False, always regenerate.

    Returns
    -------
    X : np.ndarray, shape (n_shots, n_detectors), dtype bool
        The syndrome data. Each row is one shot; each column is one detector
        (one stabilizer at one round). Detector ordering follows Stim's
        canonical (round, stabilizer-index) layout.
    y : np.ndarray, shape (n_shots,), dtype bool
        The logical-flip labels. y[i] = True means a logical error occurred
        in shot i.
    dem : stim.DetectorErrorModel
        The decomposed detector error model for this circuit. Needed by
        PyMatching for the MWPM baseline. NOT cached — rebuilt from the
        circuit each call (it's cheap).
    """
    if distance % 2 == 0:
        raise ValueError(f"distance must be odd; got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1; got {rounds}")
    if not (0.0 < noise_p < 1.0):
        raise ValueError(f"noise_p must be in (0, 1); got {noise_p}")
    if n_shots < 1:
        raise ValueError(f"n_shots must be >= 1; got {n_shots}")

    circuit = _build_circuit(distance, rounds, noise_p, code_task)
    dem = circuit.detector_error_model(decompose_errors=True)

    cache_path = _cache_path(distance, rounds, noise_p, n_shots, seed, code_task)
    if use_cache and cache_path.exists():
        data = np.load(cache_path)
        X = data["X"]
        y = data["y"]
        if X.shape[0] == n_shots:
            return X, y, dem
        # Cache exists but wrong shape — fall through and regenerate.

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=n_shots, separate_observables=True)
    # dets: (n_shots, n_detectors) bool
    # obs:  (n_shots, n_observables) bool — 1 observable for memory experiments
    X = np.asarray(dets, dtype=bool)
    y = np.asarray(obs, dtype=bool).ravel()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y, dem


def get_circuit(
    distance: int,
    rounds: int,
    noise_p: float,
    code_task: str = "surface_code:rotated_memory_z",
) -> stim.Circuit:
    """Return the Stim circuit without sampling. Useful for diagnostics."""
    return _build_circuit(distance, rounds, noise_p, code_task)


def describe_dataset(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Return a small dictionary of summary statistics for a dataset.

    Used in the EDA notebook to print clean tables.
    """
    n_shots, n_detectors = X.shape
    n_flips = int(y.sum())
    return {
        "n_shots": n_shots,
        "n_detectors": n_detectors,
        "n_logical_flips": n_flips,
        "logical_error_rate": n_flips / n_shots if n_shots else 0.0,
        "detector_firing_rate_mean": float(X.mean()),
        "detector_firing_rate_min": float(X.mean(axis=0).min()),
        "detector_firing_rate_max": float(X.mean(axis=0).max()),
        "fraction_clean_shots": float((X.sum(axis=1) == 0).mean()),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_circuit(
    distance: int,
    rounds: int,
    noise_p: float,
    code_task: str,
) -> stim.Circuit:
    """Build the canonical Stim surface-code circuit with circuit-level noise."""
    return stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise_p,
        after_reset_flip_probability=noise_p,
        before_measure_flip_probability=noise_p,
        before_round_data_depolarization=noise_p,
    )


def _cache_path(
    distance: int,
    rounds: int,
    noise_p: float,
    n_shots: int,
    seed: int,
    code_task: str,
) -> Path:
    """Build a deterministic cache filename from the generation parameters."""
    # Short hash of the code task so the filename stays readable.
    task_hash = hashlib.md5(code_task.encode()).hexdigest()[:6]
    name = (
        f"d{distance}_r{rounds}_p{noise_p:.0e}_n{n_shots}_s{seed}_{task_hash}.npz"
    )
    return CACHE_DIR / name


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running smoke test...")
    X, y, dem = generate_dataset(
        distance=3, rounds=3, noise_p=1e-3, n_shots=10_000, seed=42
    )
    info = describe_dataset(X, y)
    print(f"  shots:               {info['n_shots']}")
    print(f"  detectors per shot:  {info['n_detectors']}")
    print(f"  logical flips:       {info['n_logical_flips']}")
    print(f"  logical error rate:  {info['logical_error_rate']:.4%}")
    print(f"  mean firing rate:    {info['detector_firing_rate_mean']:.4%}")
    print(f"  clean shots:         {info['fraction_clean_shots']:.2%}")
    print(f"  DEM error count:     {dem.num_errors}")
    print("OK")