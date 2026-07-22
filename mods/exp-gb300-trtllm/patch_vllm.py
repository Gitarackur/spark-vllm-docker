#!/usr/bin/env python3
"""Patch vLLM to stage marked UVA-offloaded TRTLLM MoE weights in HBM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn


PREFIX = "[exp-gb300-trtllm]"
MARKER = "exp-gb300-trtllm: stage marked TRTLLM MoE UVA weights in HBM"
TARGET_RELATIVE_PATH = Path("vllm/model_executor/model_loader/utils.py")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"{PREFIX} {message}")


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(
            f"expected exactly one source anchor for {description}, found {count}; "
            "refusing to patch an unknown vLLM source layout"
        )
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: patch_vllm.py SITE_PACKAGES")

    site_packages = Path(sys.argv[1]).resolve()
    target = site_packages / TARGET_RELATIVE_PATH
    if not target.is_file():
        fail(f"vLLM source file not found: {target}")

    original = target.read_text()
    if MARKER in original:
        print(f"{PREFIX} Patch is already applied; skipping.")
        return

    text = original

    text = replace_once(
        text,
        """            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
""",
        f"""            # {MARKER}
            # A CUDA UVA view reports itself as a CUDA tensor, so the generic
            # loading context otherwise leaves these large, randomly indexed
            # tensors in host LPDDR during TRTLLM layout conversion.
            nvfp4_backend = getattr(quant_method, "nvfp4_backend", None)
            stage_trtllm_moe_uva = (
                getattr(nvfp4_backend, "name", None) == "FLASHINFER_TRTLLM"
            )
            with device_loading_context(
                module,
                target_device,
                stage_trtllm_moe_uva=stage_trtllm_moe_uva,
            ):
                quant_method.process_weights_after_loading(module)
""",
        "quantized post-load device context",
    )

    text = replace_once(
        text,
        """@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
""",
        """@contextmanager
def device_loading_context(
    module: torch.nn.Module,
    target_device: torch.device,
    *,
    stage_trtllm_moe_uva: bool = False,
):
""",
        "device_loading_context signature",
    )

    text = replace_once(
        text,
        """    original_device_states: dict[str, torch.device] = {}
    uva_offloaded_parameters: list[str] = []

    # Store original device states and move parameters to GPU if they're on CPU
""",
        """    original_device_states: dict[str, torch.device] = {}
    uva_offloaded_parameters: list[str] = []
    staged_uva_backings: list[torch.Tensor] = []
    staged_uva_bytes = 0
    staged_uva_count = 0

    # Keep the original UVA views alive until the converted HBM tensors have
    # been copied back to new pinned CPU storage in the finally block.
    # Store original device states and move parameters to GPU if they're on CPU.
""",
        "UVA staging state",
    )

    text = replace_once(
        text,
        """        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
        # Parameters already on target device are not touched

    try:
""",
        """        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
            if stage_trtllm_moe_uva:
                # A same-device .to() is a no-op for a UVA CUDA view. clone()
                # allocates fresh CUDA storage, which is HBM on GB300, and copies
                # exactly the parameters vLLM actually offloaded, including a
                # partially offloaded final module, through the C2C link once.
                staged_uva_backings.append(p.data)
                p.data = p.data.clone(memory_format=torch.preserve_format)
                delattr(p, "_vllm_is_uva_offloaded")
                staged_uva_bytes += p.numel() * p.element_size()
                staged_uva_count += 1
        # Parameters already on target device are not touched.

    if staged_uva_bytes:
        logger.info(
            "Staged %d UVA-offloaded TRTLLM NVFP4 MoE parameters (%.2f GiB) "
            "in HBM for post-load layout conversion.",
            staged_uva_count,
            staged_uva_bytes / (1024**3),
        )

    try:
""",
        "marker-driven UVA-to-HBM staging loop",
    )

    # The existing finally block re-offloads a parameter whenever its original
    # name was UVA-backed and the current Parameter no longer carries the UVA
    # marker. Staging deliberately removes that marker, so both in-place and
    # replacement-style quantization paths are restored without another change.

    try:
        compile(text, str(target), "exec")
    except SyntaxError as exc:
        fail(f"patched source failed Python syntax validation: {exc}")

    target.write_text(text)

    print(
        f"{PREFIX} Applied marker-driven HBM staging for UVA-offloaded "
        "TRTLLM NVFP4 MoE post-load conversion."
    )


if __name__ == "__main__":
    main()
