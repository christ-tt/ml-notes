---
tags:
  - ML
  - Infra
  - Work
---

### Do we “keep accepting” new requests in prefill-favor continuous batching?
Yes—new arrivals keep going into `waiting_queue`, and on each scheduling step the scheduler tries to **form a new prefill batch from the waiting queue** (even while there’s an existing decode `running_batch`). But it will **stop admitting** once any of its admission budgets says “no”.

The core loop is in `Scheduler.get_new_batch_prefill()`:

```1662:1755:/Users/senyutong/working/sglang/python/sglang/srt/managers/scheduler.py
def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    # ...
    adder = PrefillAdder(
        self.tree_cache,
        self.token_to_kv_pool_allocator,
        self.running_batch,
        self.new_token_ratio,
        self.max_prefill_tokens,
        self.chunked_prefill_size,
        running_bs if self.is_mixed_chunk else 0,
    )

    # Get requests from the waiting queue to a new prefill batch
    for req in self.waiting_queue:
        # ... microbatch / lora / pp limits ...
        req.init_next_round_input(...)
        res = adder.add_one_req(req, self.chunked_req, self.enable_hierarchical_cache)
        if res != AddReqResult.CONTINUE:
            if res == AddReqResult.NO_TOKEN:
                self.running_batch.batch_is_full = True
            break
```

### What heuristic stops admitting new prefills “early” (given unknown output length)?
The heuristic is: **reserve KV for an estimate of each request’s future total tokens**, and stop admitting when the reservation would exceed KV capacity (plus whatever can be evicted from prefix cache).

That’s implemented in `PrefillAdder`:

- **KV budget check**: stop if `total_tokens >= rem_total_tokens`
- **Per-batch prefill-token budget**: stop when you’ve used up `max_prefill_tokens` (and optionally `chunked_prefill_size`)

```268:335:/Users/senyutong/working/sglang/python/sglang/srt/managers/schedule_policy.py
class PrefillAdder:
    def __init__(...):
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
        # ...

    @property
    def rem_total_tokens(self):
        return (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
            - self.rem_total_token_offset
        )

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN
        if self.rem_input_tokens <= 0 or (...):
            return AddReqResult.OTHER
        return AddReqResult.CONTINUE
```

For a *new* request, the scheduler uses **“prompt tokens + estimated max_new_tokens”**, where the estimate is **clipped**:

```445:497:/Users/senyutong/working/sglang/python/sglang/srt/managers/schedule_policy.py
total_tokens = req.extend_input_len + min(
    req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
)
input_tokens = (
    -(-req.extend_input_len // self.tree_cache.page_size) * self.tree_cache.page_size
)

if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN

# ... then it consumes input_tokens (page-aligned) and reserves clipped max_new_tokens ...
self._prefill_one_req(prefix_len, input_tokens, min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION))
return self.budget_state()
```

So when you “don’t know output will reach 7K”, **the scheduler still has to assume something**. It assumes up to:
- the request’s `max_new_tokens` (user-specified / default), but
- **clipped by** `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` (default 4096).

### Walk-through with your numbers (32K KV, 8K max ctx, 1K prompt, page=128)
Assume:
- KV pool holds **32K tokens**
- page size = **128**
- each request prompt length = **1000**
- max context len = **8000**, so worst-case output could be **7000**, but admission uses `min(max_new_tokens, 4096)`.

**Page-aligned prompt KV cost**
- \( \lceil 1000/128 \rceil \times 128 = 1024 \) KV “tokens” reserved for the prompt.

**Estimated output KV cost**
- If `max_new_tokens` is 7000, admission uses `min(7000, 4096) = 4096`.

**Total KV reserved per admitted request (for admission control)**
- \( 1024 + 4096 = 5120 \) KV tokens (i.e., \(5120/128 = 40\) pages).

**How many requests can be admitted before we stop?**
- \( 32000 / 5120 \approx 6.25 \) ⇒ you can admit **6 requests**, then the 7th fails the `NO_TOKEN` check and admission stops for that prefill batch.

### One more nuance: “conservativeness” for existing in-flight decode
When there’s already a `running_batch`, `PrefillAdder` also subtracts an estimate of **remaining decode growth** from the available KV budget using `new_token_ratio` (derived from `schedule_conservativeness`):

```406:428:/Users/senyutong/working/sglang/python/sglang/srt/managers/scheduler.py
self.init_new_token_ratio = min(
    global_config.default_init_new_token_ratio * server_args.schedule_conservativeness,
    1.0,
)
self.new_token_ratio = self.init_new_token_ratio
```

Meaning: if you already have in-flight decode, the scheduler becomes **more conservative about admitting new prefills**, because it expects the running requests to keep consuming KV.

### Bottom line
- **We do keep accepting arrivals into the waiting queue**, but we **stop admitting prefills** into the next prefill batch once KV/page budget (and/or per-batch prefill token budget) would be exceeded.
- For unknown output length, the main heuristic is **reserve using `max_new_tokens` (clipped to 4096 by default)**, plus **page-aligned prompt tokens**; this is exactly the “stop admitting early” mechanism you asked about.
