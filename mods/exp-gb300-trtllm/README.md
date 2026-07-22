# Experimental GB300 TRTLLM startup mod

This runtime mod accelerates the post-load TensorRT-LLM NVFP4 MoE weight-layout
conversion when selected expert weights have been offloaded with vLLM's UVA
backend on a Grace-Blackwell system.

vLLM normally sees a UVA mapping as a CUDA tensor and therefore performs the
TRTLLM expert permutation directly against host LPDDR. This mod detects the
`FLASHINFER_TRTLLM` NVFP4 MoE backend and uses vLLM's internal
`_vllm_is_uva_offloaded` parameter marker to clone every parameter that was
actually UVA-offloaded for the current MoE layer into HBM. It runs the existing
conversion unchanged, then lets vLLM copy the converted tensors back to pinned
LPDDR before proceeding to the next layer.

The internal marker is the source of truth after vLLM applies both
`--cpu-offload-params` and `--cpu-offload-gb`. Consequently, the mod does not
assume names such as `w13_weight` or `w2_weight`; it also handles broader
selectors such as `experts`, future parameter names, and a final layer that is
only partially offloaded because the byte budget was reached.

The staging is bounded to one MoE layer at a time. It does not change runtime
weight placement, KV-cache placement, model output, or the selected inference
kernel. It does temporarily increase load-time HBM usage by the staged layer
plus TRTLLM conversion intermediates.

Apply it like the other spark-vllm mods:

```bash
./launch-cluster.sh --solo \
  --apply-mod mods/exp-gb300-trtllm \
  exec vllm serve nvidia/GLM-5.2-NVFP4 ...
```

The patch targets the source layout used by vLLM 0.25.1 and fails closed if its
expected anchors are not found. It is idempotent. During model loading, each
staged layer emits a line similar to:

```text
Staged 2 UVA-offloaded TRTLLM NVFP4 MoE parameters (4.50 GiB) in HBM for post-load layout conversion.
```

If loading runs out of HBM before KV-cache allocation, remove this mod; do not
raise `--cpu-offload-gb` merely to compensate, because that does not increase
the temporary HBM headroom available during conversion.
