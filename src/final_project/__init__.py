"""Top level package for the unified AI‑driven video cutter.

This package combines the algorithmic strengths of two independent projects:

* An AI powered slicer/resizer capable of speech driven segmentation,
  beat detection and intelligent selection of video fragments.
* A modular short‑video generator that offers flexible cropping,
  resizing and a decorator based overlay system for blur, stickers and sound.

The unified project adds automatic subtitle generation and is designed
for CUDA acceleration on modern NVIDIA GPUs.  See the `README.md`
in the project root for usage instructions.
"""

__all__ = [
    "core",
    "segmentation",
    "decorators",
    "generator",
    "gpu",
    "models",
    "gui",
]
