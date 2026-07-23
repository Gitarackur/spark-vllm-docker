# EAGLE Fine Prefix Fix

**Last updated:** `2026-07-23T15:24:29-07:00`

This runtime-only mod restores fine-grained hybrid prefix-cache hits when MTP
or EAGLE is enabled on full-attention + Mamba models such as Qwen 3.5/3.6.
It targets current vLLM after
[#46384](https://github.com/vllm-project/vllm/pull/46384), based on
`41ea2dd44a3a20c46ebeb985de0022c7673fb953`.

## Why it helps

PR #46384 registers only the final prompt-tail hash at `prefix_match_unit`
granularity. A context-load probe such as llama-benchy's `"."` can occupy that
last hash unit, so a follow-up user turn misses the entry even though the long
system context is identical. Under EAGLE, full attention then rewinds the next
available match, while Mamba falls back to the previous physical state block
(2,144 tokens for this Qwen model).

This mod also retains the predecessor FullAttention hash. Since EAGLE rewinds
that match by one more hash unit, the scheduler materializes and caches the
Mamba state at the resulting replay boundary. The extra metadata/state cost is
constant per prompt rather than changing physical allocation geometry.

## Related upstream work

The following status is a point-in-time snapshot from the last-updated
timestamp above. No searched upstream PR currently implements this mod's exact
two-part mechanism: retaining the predecessor `prefix_match_unit`
FullAttention hash and materializing the corresponding Mamba replay snapshot
after the EAGLE rewind.

- [#49574](https://github.com/vllm-project/vllm/pull/49574), an open
  experimental draft, is the closest conceptual match. It evaluates explicit
  input-end and decode-end recurrent checkpoints to avoid first-continuation
  misses when generation-only prompt suffixes make the literal prompt-end
  checkpoint unreachable. It requires structured frontend checkpoint metadata
  rather than automatically retaining the predecessor prompt hash.
- [#48815](https://github.com/vllm-project/vllm/pull/48815) is the closest
  compact performance fix. It conditionally avoids EAGLE's full physical-block
  backoff for an MTP prompt with an uncached tail and reports a substantial
  hot-cache TTFT improvement. It does not add the predecessor fine-grained
  FullAttention hash or partial Mamba replay snapshot. The PR had merge
  conflicts at the snapshot timestamp.
- Merged [#46384](https://github.com/vllm-project/vllm/pull/46384) provides the
  partial-prefix infrastructure on which this mod builds, but only registers
  the final prompt-tail hash.
- Open [#45614](https://github.com/vllm-project/vllm/pull/45614) and
  [#46281](https://github.com/vllm-project/vllm/pull/46281) address unsafe or
  mismatched EAGLE/Mamba cache hits. They are correctness fixes and do not
  retain the extra fine-grained changed-suffix checkpoint.
- [#47861](https://github.com/vllm-project/vllm/pull/47861) was a broader
  hybrid MTP prefix-cache correctness attempt and closed without merging.

## Usage

Pass the match unit after the recipe runner's `--` separator:

```bash
./run-recipe.sh \
  --apply-mod mods/fix-eagle-fine-prefix \
  --solo recipes/qwen3.6-35b-a3b-nvfp4.yaml \
  -- --prefix-match-unit 16
```

Do not combine this with `mods/pr-46251-hybrid-large-blocks`. The older mod
changes physical allocation geometry; this one keeps current vLLM's physical
blocks and fixes the missing fine-grained Mamba snapshot.

At startup, verify the `cache_config_info` metric reports
`prefix_match_unit="16"`. For performance tests, compare the delta in
`vllm:prefix_cache_hits_total` as well as TTFT. A successful warm request should
gain hits near the shared-prefix length instead of a multiple of 2,144.

## Validation

The change was tested against current vLLM's partial-prefix-cache suite:

```text
14 passed
```

Two added regression cases fail on unpatched vLLM and pass with the mod:

- the scheduler materializes the Mamba state at the replay point behind the
  predecessor prompt hash;
- a follow-up whose short user suffix differs from the context-load request
  retains a fine-grained hybrid hit instead of falling back 2,144 tokens.
