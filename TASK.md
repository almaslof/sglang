# NSA topk buffer + ROCm GPU memory access fault

## Goal

- **Stable baseline:** commit `7c35342c1`.
- **Symptom on ROCm:** `Memory access fault by GPU node-8 ... on address ... Reason: Unknown.`
- **Requirements:** fix the fault; keep the diff vs `7c35342c1` small; keep **preallocated** `topk_indices` scratch in **`Indexer.__init__`** (no per-forward `torch.full` on hot paths where possible).

## Fixes retained vs baseline (minimal set)

1. **`nsa_backend.py` — fused topk vs page-table transform**  
   When `SGLANG_NSA_FUSE_TOPK` is on but `NSAIndexerMetadata.force_unfused_topk` is also true (e.g. HiSparse + decode), the indexer uses **logical** indices (`fast_topk_v2`). The backend must **not** skip `transform_index_page_table_{prefill,decode}` in that case, or sparse gather uses wrong indices → GPU fault.

2. **`Indexer` — where to preallocate**  
   Preallocation must live in **`Indexer.__init__`**, not as **one tensor shared by every layer** in `DeepseekV2Model`.

## Root cause hypothesis (shared buffer across layers)

NSA **index-cache** (`skip_topk` / `next_skip_topk` in `forward_mla.py`) can set `topk_indices = prev_topk_indices` so multiple layers **alias** the same tensor storage.

If **all** layers also shared **one** `topk_indices_buffer`, a later layer’s indexer could run **`fill_(-1)`** on that storage while earlier layers’ returned tensors still pointed at it. Even with normal GPU stream ordering this is fragile; on ROCm it showed up as a **memory access fault**.

**Mitigation:** each **`Indexer`** instance allocates its **own** `(max_prefill_tokens, index_topk)` int32 buffer in `__init__` (unless tests inject `topk_indices_buffer`). Layers no longer stomp each other’s scratch.

**Trade-off:** memory scales with **number of NSA layers** × buffer size (vs one shared buffer). Typical order: tens of MB per layer slot depending on `max_prefill_tokens` and `index_topk`.

## If the fault persists — bisection checklist

1. **Confirm** `SGLANG_NSA_FUSE_TOPK` + `force_unfused_topk` path: temporarily set `SGLANG_NSA_FUSE_TOPK=0` — if fault disappears, focus on `transform_index` / fused vs unfused contract.
2. **Batch size vs `max_prefill_tokens`:** if any forward uses more tokens than `server_args.max_prefill_tokens`, `_get_topk_result_buffer` falls back to a fresh `torch.full`; ensure that path is not hitting a bad kernel shape.
3. **Enable sync validation (dev only):** add a short-lived `torch.cuda.synchronize()` before/after indexer in the failing scenario to see if it’s async ordering (narrows race vs bad indices).
4. **Compare** `git diff 7c35342c1` — keep only `nsa_backend.py` gating + `Indexer` buffer + `nsa_indexer` helper changes.

## Files touched (intended)

- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` — `Indexer.__init__` prealloc + `_get_topk_result_buffer`
- `python/sglang/srt/layers/attention/nsa_backend.py` — `force_unfused_topk` + fused-topk gating
- `python/sglang/srt/models/deepseek_v2.py` — **no** shared buffer; `Indexer` owns allocation
