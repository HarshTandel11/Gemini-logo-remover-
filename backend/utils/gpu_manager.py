"""
GPU memory management utilities.

Provides :class:`GPUManager` – a lightweight registry that tracks loaded
models, logs CUDA memory statistics, and exposes a context-manager
(:meth:`stage`) for profiling pipeline stages.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator

import torch

logger = logging.getLogger(__name__)


class GPUManager:
    """Manage GPU model lifecycle and memory reporting.

    Models are registered by *name* so they can be individually unloaded
    when no longer needed, freeing VRAM for the next pipeline stage.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}

    # ── Model lifecycle ───────────────────────────────────────────────────

    def load_model(self, name: str, load_fn: Callable[[], Any]) -> Any:
        """Load a model via *load_fn*, register it under *name*, and return it.

        Args:
            name: Human-readable identifier used for logging and later
                  retrieval / unloading.
            load_fn: Zero-argument callable that returns the loaded model.

        Returns:
            The model object returned by *load_fn*.
        """
        logger.info("Loading model: %s", name)
        self.log_memory(f"before_{name}")
        model = load_fn()
        self._models[name] = model
        self.log_memory(f"after_{name}")
        return model

    def unload_model(self, name: str) -> None:
        """Remove a previously registered model and release its VRAM.

        Args:
            name: The identifier used during :meth:`load_model`.
        """
        if name in self._models:
            logger.info("Unloading model: %s", name)
            del self._models[name]
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory(f"freed_{name}")
        else:
            logger.warning("Model '%s' not found in registry – nothing to unload.", name)

    def unload_all(self) -> None:
        """Unload every registered model."""
        for name in list(self._models.keys()):
            self.unload_model(name)

    # ── Memory introspection ──────────────────────────────────────────────

    def get_memory_stats(self) -> Dict[str, float]:
        """Return current CUDA memory stats in megabytes.

        Returns:
            A dict with keys ``allocated_mb``, ``reserved_mb``, and
            ``total_mb``.  All values are ``0`` when CUDA is unavailable.
        """
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }

    def log_memory(self, label: str = "") -> None:
        """Emit an INFO log line with the current GPU memory breakdown.

        Args:
            label: Optional tag that is prepended to the log message.
        """
        stats = self.get_memory_stats()
        logger.info(
            "[GPU %s] Allocated: %.0f MB, Reserved: %.0f MB",
            label,
            stats["allocated_mb"],
            stats["reserved_mb"],
        )

    # ── Context manager for pipeline stages ───────────────────────────────

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager that logs GPU memory before and after a stage.

        Usage::

            with gpu_manager.stage("inpainting"):
                run_inpainting(...)

        Args:
            name: Stage name used in the log messages.
        """
        self.log_memory(f"{name}_start")
        try:
            yield
        finally:
            self.log_memory(f"{name}_end")
