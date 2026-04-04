"""
GPU Setup Utility
=================
Centralised TensorFlow GPU configuration for all pipelines.

Call `configure_gpu()` early — before building any models — to:
  - Enable memory-growth so TF doesn't grab all VRAM at once
  - Optionally cap the per-GPU memory limit
  - Print a clear summary of what devices are available

Works gracefully on CPU-only installs (native Windows) and
GPU-capable environments (WSL2 / Linux with CUDA).
"""

import os
import warnings

# Suppress noisy TF logs before import
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import tensorflow as tf


def configure_gpu(memory_limit_mb=None, verbose=True):
    """Configure TensorFlow GPU settings.

    Parameters
    ----------
    memory_limit_mb : int or None
        If set, cap each GPU to this many MB of VRAM.
        If None, enable memory-growth instead (allocate as needed).
    verbose : bool
        Print device summary.

    Returns
    -------
    list[tf.config.PhysicalDevice]
        The list of available GPU devices (empty on CPU-only systems).
    """
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                if memory_limit_mb is not None:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=memory_limit_mb
                        )],
                    )
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Devices must be configured before TF runtime init
                if verbose:
                    print(f"  [gpu_setup] Warning: {e}")

    if verbose:
        if gpus:
            names = [g.name for g in gpus]
            mode = (f"memory limit {memory_limit_mb} MB"
                    if memory_limit_mb else "memory-growth enabled")
            print(f"  [gpu_setup] {len(gpus)} GPU(s) detected: {names}  ({mode})")
        else:
            print("  [gpu_setup] No GPU detected — running on CPU.")

    return gpus
