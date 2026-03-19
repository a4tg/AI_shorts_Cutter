"""Helpers for GPU capability detection and runtime diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from functools import lru_cache
from typing import List


@dataclass(frozen=True)
class RuntimeDiagnostics:
    torch_installed: bool
    torch_cuda_available: bool
    torch_version: str
    torch_cuda_version: str
    gpu_name: str
    nvidia_smi_available: bool
    ffmpeg_available: bool
    ffmpeg_nvenc_available: bool
    execution_device: str
    recommendations: List[str]


def _run_command(command: List[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return False, ""
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return completed.returncode == 0, output.strip()


@lru_cache(maxsize=1)
def ffmpeg_nvenc_available() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    ok, output = _run_command(["ffmpeg", "-encoders"])
    if not ok and not output:
        return False
    return "h264_nvenc" in output or "hevc_nvenc" in output


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def probe_runtime_diagnostics() -> RuntimeDiagnostics:
    torch_installed = False
    torch_cuda = False
    torch_version = ""
    torch_cuda_version = ""
    gpu_name = ""

    try:
        import torch

        torch_installed = True
        torch_version = str(getattr(torch, "__version__", ""))
        torch_cuda = bool(torch.cuda.is_available())
        torch_cuda_version = str(getattr(torch.version, "cuda", "") or "")
        if torch_cuda:
            try:
                gpu_name = str(torch.cuda.get_device_name(0))
            except Exception:
                gpu_name = ""
    except Exception:
        pass

    nvidia_smi_available = shutil.which("nvidia-smi") is not None
    if not gpu_name and nvidia_smi_available:
        ok, output = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if ok:
            gpu_name = output.splitlines()[0].strip()

    ffmpeg_available = shutil.which("ffmpeg") is not None
    ffmpeg_nvenc_is_available = ffmpeg_nvenc_available() if ffmpeg_available else False

    recommendations: List[str] = []
    execution_device = "cuda" if torch_cuda and ffmpeg_nvenc_is_available else "cpu"

    if not ffmpeg_available:
        recommendations.append("Install FFmpeg and add it to PATH.")
    elif nvidia_smi_available and not ffmpeg_nvenc_is_available:
        recommendations.append("FFmpeg is installed but NVENC is unavailable. Install an FFmpeg build with NVENC support.")

    if nvidia_smi_available and not torch_cuda:
        recommendations.append("NVIDIA GPU detected, but CUDA PyTorch is unavailable. Install a CUDA-enabled torch build.")
    if not torch_installed:
        recommendations.append("PyTorch is not installed. Install project dependencies before running the app.")
    if nvidia_smi_available and not gpu_name:
        recommendations.append("NVIDIA GPU tools were detected, but GPU name could not be resolved. Check the NVIDIA driver installation.")

    return RuntimeDiagnostics(
        torch_installed=torch_installed,
        torch_cuda_available=torch_cuda,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        gpu_name=gpu_name,
        nvidia_smi_available=nvidia_smi_available,
        ffmpeg_available=ffmpeg_available,
        ffmpeg_nvenc_available=ffmpeg_nvenc_is_available,
        execution_device=execution_device,
        recommendations=recommendations,
    )


def format_runtime_summary(diagnostics: RuntimeDiagnostics) -> List[str]:
    gpu_label = diagnostics.gpu_name or ("NVIDIA GPU detected" if diagnostics.nvidia_smi_available else "No GPU detected")
    lines = [
        f"Runtime device: {diagnostics.execution_device.upper()}",
        f"PyTorch: {diagnostics.torch_version or 'not installed'}",
        f"CUDA available: {'yes' if diagnostics.torch_cuda_available else 'no'}",
        f"GPU: {gpu_label}",
        f"FFmpeg: {'yes' if diagnostics.ffmpeg_available else 'no'}",
        f"NVENC: {'yes' if diagnostics.ffmpeg_nvenc_available else 'no'}",
    ]
    for recommendation in diagnostics.recommendations:
        lines.append(f"Recommendation: {recommendation}")
    return lines
