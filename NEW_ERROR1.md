# NEW_ERROR1 — MoE weight load / `fused_moe_triton` thread stacks

## What showed up

Logs (see also `NEXT_ERROR.md`) show many worker threads with stacks ending in:

- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- `_load_w13` / `_load_w2` → `expert_data.copy_(loaded_weight)`
- `_weight_loader_impl` / `_weight_loader_physical` / `weight_loader`

The log snippet is interleaved/corrupted; the underlying failure is often **GPU OOM** or **shape mismatch** during `copy_`, not a bug in the copy line itself.

## Likely cause after per-layer NSA `topk_indices_buffer`

Allocating **one large int32 buffer per NSA layer** in `Indexer.__init__` runs **during model construction**, **before** MoE expert weights finish loading. That increases **peak device memory** at the worst time for large models.

## Fix applied

**Lazy** allocation: `self.topk_indices_buffer = None` in `__init__` when no buffer is injected. The tensor is created on the **first** `_get_topk_result_buffer()` call (first real forward), using the runtime `device` for that pass.

Behavior preserved:

- Still **one buffer per `Indexer`** (safe with `skip_topk` / `prev_topk_indices` aliasing).
- Still **no per-forward `torch.full`** on the hot path after the first forward.

---

If errors persist, capture the **root exception** above the thread dumps (e.g. `torch.cuda.OutOfMemoryError` or `RuntimeError: size mismatch`).
