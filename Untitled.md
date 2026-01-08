# 0. Overview

In this tech blog, we will systematically analyze the performance of enabling edge devices hosting large language models, with Pipeline Parallel. And compare the performance under different hardwares and p2p latency.

# **1. Decode - Memory Bandwidth Bottleneck**

To understand why decode is bottlenecked by memory bandwidth rather than compute, we start from the basic cost model of a **GEMM** (general matrix multiplication).

## **1.1 GEMM cost model**

Consider a matmul:

$A \in \mathbb{R}^{M\times K},\quad B \in \mathbb{R}^{K\times N},\quad C = A B \in \mathbb{R}^{M\times N}.$

### **Compute (FLOPs)**

A standard GEMM does, for each output element $C_{ij}$,

- K multiplies and K-1 adds ≈ 2K FLOPs,

so the total FLOPs is:

$\text{FLOPs} \approx 2 M K N$.

### **IO (HBM ↔ on-chip SRAM)**

Ignoring caching subtleties and assuming we stream each matrix once:

- Load $A$: $M K$ elements
- Load $B$: $K N$ elements
- Store $C$: $M N$ elements

For bf16 (2 bytes per element), total traffic is:

$\text{Bytes} \approx 2 \cdot (M K + K N + M N)$.

If the GPU peak compute is $\text{TFLOPs}$ (in FLOPs/s) and HBM bandwidth is $\text{BW}$ (in bytes/s), then:

- Compute-limited time:

$t_{\text{comp}} \approx \frac{2 M K N}{\text{TFLOPs}}$.

- Memory-limited time:

$t_{\text{io}} \approx \frac{2 (M K + K N + M N)}{\text{BW}}$.

Whichever is larger dominates the actual latency.

---

## **1.2 Why Focus on GEMM (and Specifically MLP / Attention GEMMs)?**

Before we dive into numbers, it’s worth asking two questions:

1. Why is **GEMM** (general matrix multiplication) the core primitive in deep learning?
2. Why do we focus on **MLP GEMMs** and **attention GEMMs** in particular?

Modern accelerators (GPUs, TPUs, Apple/ANE-style chips) are built to run **dense matrix multiplications** as fast as possible. A Transformer layer is mostly linear algebra: Q/K/V projections, attention output projections, and the linear layers in the MLP block. All of these can be written as GEMMs of the form

$[M, K] \times [K, N] \to [M, N]$. From the hardware’s perspective, “running a model” is essentially “running a sequence of GEMMs plus some relatively cheap elementwise ops.” That’s why Tensor Cores, cuBLAS, CUTLASS, etc. all obsess over GEMM performance.

Within a decoder layer, most meaningful compute and memory traffic comes from two families of GEMMs:

- **MLP GEMMs**
    
    The feed-forward block uses large projections like
    
    $[B\cdot S, d_{\text{model}}] \times [d_{\text{model}}, d_{\text{ff}}]$ (and the reverse).
    
    These multiply big activation batches by **very large weight matrices**. They dominate **parameter size** and **FLOPs**, and their behavior flips between **compute-bound** (prefill, long sequences) and **memory-bound** (decode, small S) depending on batch and sequence length.
    
- **Attention GEMMs**
    
    Attention consists of:
    
    - Q/K/V projections (again, GEMMs similar to the MLP projections),
        
    - the **score** computation $QK^\top$,
        
    - and the **value mixing** $\text{softmax}(QK^\top)V$.
        
        These are where **KV cache** and **sequence length** bite hardest. The score/value GEMMs touch $O(S^2)$, and tend to be **memory-bandwidth-bound**, especially in decode.
        

In the rest of this section, we therefore zoom in on:

- the **MLP GEMM** as the representative “big weight matrix” compute, and
- the **attention GEMMs** (QKᵀ and softmax·V) as the representative “KV-cache-heavy” compute.

---

## **1.3 Concrete setup: Qwen2-32B on RTX 4090 (decode vs prefill)**

Let’s plug numbers for a 32B-class dense model (Qwen2-32B) on a single RTX 4090:

- Model:
    - Hidden size $d_{\text{model}} = 5120$
    - MLP intermediate size $d_{\text{ff}} = 27{,}648$
    - Query heads $H_q = 40$
    - KV heads $H_{kv} = 8$ (GQA)
    - Per-head dim $d_h = 5120 / 40 = 128$
- Hardware (4090 approximate):
    - Tensor Core FP16/BF16 peak ≈ $\text{TFLOPs}$ = $165 \times 10^{12}$ FLOPs/s
    - Memory bandwidth ≈ $\text{BW}$ = $1001 \times 10^{9}$ bytes/s
- Precision: bf16 ⇒ 2 bytes/element
- Batch size $B = 16$

---

## 1.4 MLP GEMM

Activation shape: $A = [B, S_{q}, d_{model}]$

Weight shape: $B: [d_{model}, d_{ff}]$

So $M = B\cdot S_{q}, K = d_{model}, N = d_{ff}$

For compute, we have

- $\text{FLOPs} = 2 M K N = 2 \cdot B \cdot S_q \cdot d_{model} \cdot d_{ff}$

for IO, we have

- $\text{Bytes} = N_{bytes}(MK+KN+MN)=2(B\cdot S_q \cdot d_{model} + d_{model}\cdot d_{ff} + B\cdot S_q \cdot d_{ff})$

### **Decode**

**We have $S_q = 1$, and we have KV cache in use, so**

$t_{\text{comp}} = \frac{2 \cdot 16 \cdot 5120 \cdot 27648} {165\times 10^{12}} \approx 2.7 \times 10^{-5}\,\text{s} = \mathbf{0.027\ \text{ms}}$

$t_{\text{io}} = \frac{2 \cdot (16\cdot 5120 + 5120 \cdot 27648 + 16\cdot 27648)}{1001 \times 10^9} \approx 2.8 \times 10^{-4}\,\text{s} = \mathbf{0.28\ \text{ms}}$

⇒ The GEMM is clearly **memory-bandwidth-bound** (≈10× gap).

Note that increasing **batch size** B hardly changes IO time here, because:

- The dominant term is $K N = d_{\text{model}} \cdot d_{\text{ff}}$, i.e., the weight matrix.
- As you increase batch size, you reuse the same weights across more rows of A and C; the extra IO is mostly in the (much smaller) $M K$ and $M N$ terms.

This is why, for MLP/linear layers in decode, **larger batch sizes improve utilization** substantially while per-token latency barely moves: you amortize the big weight load over more tokens.

### **Prefill**

Now with context length S = 1024:

$t_{\text{comp}} = \frac{2 \cdot 16 \cdot 1024\cdot 5120 \cdot 27648} {165\times 10^{12}} \approx 0.028\,\text{s} = \mathbf{28.1\ \text{ms}}$

$t_{\text{io}} = \frac{2 \cdot (16\cdot 1024\cdot 5120 + 5120 \cdot 27648 + 16 \cdot 1024 \cdot 27648)}{1001 \times 10^9} \approx 1.3 \times 10^{-3}\,\text{s} = \mathbf{1.3\ \text{ms}}$

Here, the MLP prefill is **compute-bound** (compute is ≈21× larger than IO), which matches intuition: we’re doing a lot of math on the same weight matrix for a long sequence.

---

## **1.5 Attention Score GEMM**

We now look at the **attention score** GEMM $QK^\top$.

We assume GQA with:

- Query heads $H_q = 40$
- KV heads $H_{kv} = 8$
- Group size $\text{group} = G = H_q / H_{kv} = 5$
- Head dim $d_h = 128$
- Context length S
- Batch size $B = 16$
- Precision bf16, so bytes per element $N_{bytes} = 2$

We have

$Q = [B, H_{q}, S_{q}, d_h]$, and

$K = [B, H_{kv}, S_{kv}, d_h]$

We can rearrange $Q = [B, H_{kv}, G, S_q, d_h]$. For each sequence, we have $G$ query heads attending to each of the kv heads. We have GEMM per sequence, per KV head of shape

$[G, S_q, d_h] \times [d_h, S_{kv}] \Rightarrow [G \cdot S_q, S_{kv}]$

Now, accounting for all sequences in the batch and all kv heads we sharded, we have

**Compute flops**: $B \cdot H_{kv} \cdot 2 \cdot ((G \cdot S_q) \cdot d_h \cdot S_{kv})$ and

**IO: $B \cdot H_{kv} \cdot N_{bytes} \cdot (G\cdot S_q \cdot d_h + d_h \cdot S_{kv} + G \cdot S_q \cdot S_{kv})$**

In contrast to the MLP GEMM, notice how **batch size** B enters both the compute and IO terms here. Both compute FLOPs and IO bytes scale linearly with batch size.

Crucially, unlike the weight **GEMM with a large weight matrix**, there is **no big static weight tensor to amortize** here:

- For **MLP / linear layers** in decode, larger batch sizes can significantly improve utilization while barely affecting per-token latency, because the dominant IO is the (batch-independent) weight matrix.
- For **attention score computation**, larger batch sizes **do** increase per-step latency (both compute and memory time) roughly linearly, aside from the benefit of more parallelism (utilizing more SM cores due to available amount for sharding). You get more work to spread across SMs, but you don’t get the same “free amortization” of a giant weight matrix as in the MLP case.

### **Decode**

During decoding, as we leverage KV cache, we have one new token per sequence, attending to all cached keys, so $S_q = 1$.

Plugging in our configs, we have

$t_{comp} = \frac{16 \cdot 8 \cdot 2 \cdot (5 \cdot 1) \cdot 128 \cdot 1024)}{165 \times 10^{9}} \approx 0.001\ \text{ms}$

$t_{io} = \frac{16 \cdot 8 \cdot 2 \cdot (5 \cdot 1 \cdot 128 + 128 \cdot 1024 + 5 \cdot 1 \cdot 1024}{1001 \times 10^6} \approx 0.035 \ \text{ms}$

Note that here we use FLOPs per micro second for compute, and bytes per micro second for IO, and thus using $165 \times 10^9$ instead of $10^{12}$, and $1001 \times 10^6$ instead of $1001 \times 10^9$, repecticely.

So for attention scores in **decode**:

- Compute ≈ **0.001 ms**
- IO ≈ **0.035 ms**

⇒ The attention score GEMM is **strongly memory-bound** (≈35× gap).

### Prefill

In prefilling, we have $S_q = S_{kv} = 1024$, the context length. We now have

$t_{comp} = \frac{16 \cdot 8 \cdot 2 \cdot (5 \cdot 1024) \cdot 128 \cdot 1024)}{165 \times 10^{9}} \approx 1.04\ \text{ms}$

$t_{io} = \frac{16 \cdot 8 \cdot 2 \cdot (5 \cdot 1024 \cdot 128 + 128 \cdot 1024 + 5 \cdot 1024 \cdot 1024}{1001 \times 10^6} \approx 1.54 \ \text{ms}$

For context length

- $S=2048$, $t_{comp} \approx 4.16\ \text{ms}$, $t_{io} \approx 5.8\ \text{ms}$
- $S=4096$, $t_{comp} \approx 16.7\ \text{ms}$, $t_{io} \approx 22.3\ \text{ms}$

So even in Prefill, attention score GEMM is still primarily IO-bound, though it is more balanced than decode: compute and IO are within a small factor of each other at large $S$.

---

A similar calculation for **softmax·V** (attention weights @ V) shows the same pattern: relatively small FLOPs, sizable KV-cache IO.

In practice, FlashAttention-style kernels don’t materialize the full $S \times S$ score matrix in HBM; instead, we tile across $S$, and intermediate partial results stay on chip. However, the basic qualitative conclusion holds: **decode attention is dominated by memory traffic, specifically sequence length (KV cache) term as $B H_{kv} S d_h$ scaling with $S$.**

---

# 2. Pipeline Parallel 

When a model is too large to fit into the VRAM of a single device (e.g., a 70B model on a 24GB RTX 4090), we must split it across multiple devices. The two primary strategies are **Tensor Parallelism (TP)** and **Pipeline Parallelism (PP)**.

## 2.1 Tensor Parallelism (TP) and Its Cost
In a datacenter environment equipped with NVLink and NVSwitch (providing 600–900 GB/s bandwidth), TP is often the default choice for intra-node parallelism. However, for edge devices connected via PCIe (32 GB/s) or worse, Ethernet/LAN, TP is performance suicide.

TP splits the individual weight matrices (e.g., $W_Q, W_K, W_V, W_O, W_{up}, W_{down}$) across devices. While this perfectly balances the compute load, it requires heavy synchronization.

Specifically, in a Transformer architecture, **every single layer** requires synchronization to aggregate partial results.

- **Attention Block:** Requires an `All-Reduce` operation to sum the outputs of the attention heads.
    
- **MLP Block:** Requires an `All-Reduce` operation to sum the outputs of the feed-forward network.
    

For a model with $L$ layers, TP requires $2L$ synchronization steps per forward pass.

If we host a 32B model (approx 60 layers) on 2 GPUs via TP:

$$\text{Sync Ops} = 2 \times 60 = 120 \text{ network calls per token}.$$

On a slow interconnect (PCIe or LAN), the latency of 120 round-trips per token dominates the execution time, effectively stalling the GPU compute.


## 2.2 Pipeline Parallelism: The Edge-Friendly Alternative

PP partitions the model by layers (e.g., Device 0 holds layers 0–29, Device 1 holds layers 30–59). Data flows sequentially: Device 0 computes the intermediate activations for the first half of the model and sends them to Device 1.

The advantages for edge hardware are clear:

1. **Communication Frequency:** We only communicate at the "cut" points. For a 2-stage pipeline, we send data **once** per micro-batch (Device 0 $\to$ Device 1).
    
2. **Message Size:** While the message size (activations of shape $[B, S, d_{model}]$) is non-trivial, it is a single large burst rather than hundreds of tiny, latency-sensitive packets.
    

## 2.3 The "Naive" Pipeline and the Bubble Problem

While PP solves the bandwidth latency issue, it introduces a **utilization** issue.

Consider the simplest implementation: Naive PP without Micro-batching.

We take our maximum batch size (e.g., $B=16$) and process it as one monolithic block.

1. **Time $0 \to t$:** Device 0 processes the batch. **Device 1 is idle (0% utilization).**
    
2. **Time $t$:** Device 0 finishes and sends activations to Device 1.
    
3. **Time $t \to 2t$:** Device 1 processes the batch. **Device 0 is idle (0% utilization)** (assuming it waits for the next request or cannot process the next batch until Device 1 is clear).
    

In this "stop-and-wait" execution model, at any given moment, only one GPU is working while the other waits for data.

$$\text{System Utilization} \approx \frac{1}{N_{\text{stages}}}$$

For a 2-GPU setup, we sacrifice 50% of our theoretical peak compute to "bubbles" (idle time).

To fix this, we need **Micro-batching**—slicing the batch into smaller chunks to keep multiple devices active simultaneously. But before we calculate optimal micro-batch sizes, we must understand what fundamentally limits the speed of a pipeline.

---

## 2.4. Working with Pipeline - Slowest Stage Determines System Throughtput

Let’s explain why the critical stage alone determines the overall throughput, irrespective of how fast the other devices are.

Before we talk about GPUs or tokens, let’s define a **stage**:

> A **stage** is a chunk of work where a _worker_ is _blocked_ until that work is done. Once a stage finishes, the item moves on to the next stage (or exits the system).

### **2.4.1 Single stage: easy case**

Suppose we have a pipeline with just **one stage** that takes **50 ms** to generate a token.

- Time per token: 50 ms
    
- Throughput:
    
    $\frac{1}{50\ \text{ms}} = 20\ \text{tokens/s}$
    

### **2.4.2 Two stages: one fast, one slow**

Now we have **two stages**:

- Stage A: **50 ms** per token
- Stage B: **100 ms** per token

Timeline:

- **t = 0**: Token 1 enters Stage A.
- **t = 50 ms**: Stage A finishes Token 1 and hands it to Stage B. Stage A immediately starts Token 2.
- **t = 100 ms**: Stage A finishes Token 2, hands it to Stage B’s queue, starts Token 3.
- **t = 150 ms**: Stage B finally finishes Token 1. Token 2 is already waiting. Token 3 is likely right behind.

From here on, Stage B:

- Takes **100 ms** to produce each completed token.
- Always has a backlog of tokens from Stage A.

Stage A might be blazing fast, but it can only _feed_ tokens into a queue. Stage B is the **bottleneck**: the highway only lets cars leave at Stage B’s rate:

$\text{Throughput} = \frac{1}{100\ \text{ms}} = 10\ \text{tokens/s}$

No matter how much we speed up Stage A, **Stage B still emits one token every 100 ms**. That’s our system throughput.

### **2.4.3 Deep pipelines: same story**

Now, let’s scale this idea up.

Imagine **every stage takes 50 ms**, and we have:

- 2 stages
- 10 stages
- 200 stages

In all cases, assume:

- There’s always more work waiting (infinite backlog).
- We ignore the initial warm-up and the final drain (tail latency).

After warm-up, the pipeline behaves like this:

- Stage 1 is working on token _i + (S−1)_
- Stage 2 is working on token _i + (S−2)_
- …
- Stage S is working on token _i_

Every **50 ms**, _each_ stage hands its token to the next stage and pulls a new one from its input queue. So:

- **Every 50 ms, the last stage finishes exactly one token.**
- That means one finished token comes out every 50 ms → **20 tokens/s**.

Crucially:

- **Latency** increases with more stages (you wait longer from input to output).
- **Throughput** does _not_ increase: it’s still limited by the **slowest** stage’s cycle time.

If one of those 200 stages is slower (say 80 ms instead of 50 ms), that stage becomes the new bottleneck:

$\text{Throughput} = \frac{1}{80\ \text{ms}} = 12.5\ \text{tokens/s}$

All other stages are forced to idle part of the time so that the overall departure rate matches that slowest stage.

---

## **2.5 Micro Batching**

In the "naive" pipeline approach described earlier, we treated the entire batch as a single unit. This leads to a stop-and-wait behavior: while GPU 1 is processing the batch, GPU 2 is completely idle, waiting for data.

To fix this, we split the global batch $B$ into smaller **micro-batches** that flow through the system independently.

### **2.5.1 Global View: Keeping Workers Busy**

Consider a 2-GPU pipeline with a global batch size of 16.

Instead of sending all 16 sequences at once, we split them into 2 micro-batches of 8 sequences each.

**The Flow:**

1. **Step 1:** GPU 0 processes Micro-Batch 1 (MB1). GPU 1 is idle (unavoidable warm-up).
    
2. **Step 2:** GPU 0 passes MB1 to GPU 1.
    
    - _Crucially_, GPU 0 does not wait. It immediately begins processing **Micro-Batch 2 (MB2)**.
        
    - **Result:** Now **both** GPUs are working simultaneously. GPU 1 works on the first chunk (MB1), while GPU 0 prepares the second chunk (MB2).
        

By slicing the work, we minimize the time where workers are waiting on peers. This "pipelining" effect is the primary driver of efficiency in multi-device inference.




To minimize bubbles (idle time), inputs are split into microbatches—smaller sub-batches that flow independently, enabling concurrent execution and overlap across stages. Also, we use asynchronous threads or streams (e.g., MPI/NCCL non-blocking ops) to overlap compute with communication.

To minimize the "stop-and-wait" bubbles inherent in naive pipeline parallelism, we employ **Micro-Batching**. Instead of processing the entire global batch $B$ as a single monolithic tensor, we slice it into $M$ smaller micro-batches of size $B_{\text{micro}}$.

For a **single microbatch** on one device, the flow looks like:

1. **Compute - r**un the local layers on the incoming activations:
    $[B_{\text{micro}}, T, d_{\text{model}}] \rightarrow [B_{\text{micro}}, T, d_{\text{model}}]$.
2. **Upload -** Enqueue the output activations to be sent to next peer (**non-blocking** send).
3. **P2P (on the wire) -** the activations move over NVLink / PCIe / Ethernet / Internet to the next device.

On the **receiving** side, **download** is posted as a non-blocking receive and overlapped with decode of previous microbatches, so we do **not** include a separate $t_{\text{download}}$ in the critical path.

For throughput analysis, that leaves us with **three potentially blocking substeps per stage and per microbatch**:

- $t_{\text{decode}}$ – local forward pass (compute + HBM traffic),
- $t_{\text{upload}}$ – time to push activations onto the link,
- $t_{\text{p2p}}$ – pure network latency.

The effective per-stage time is therefore

$t_{\text{stage}} \approx \max\big( t_{\text{decode}},\ t_{\text{upload}},\ t_{\text{p2p}} \big)$, where $t_{\text{decode}} = \max(t_{\text{io}},\ t_{\text{compute}})$.



### 2.5.1 Number of Stages, Micro-Batches, and Bubbles

There’s a subtle cost to **making the pipeline very deep**: to keep all stages busy, you must slice the work into finer and finer pieces, otherwise you get **pipeline bubbles** (idle stages).

In most **one-pass streaming pipelines**, this is fine. As long as new work keeps arriving, once a stage finishes its current batch, it immediately pulls the next one from the input queue. You can keep increasing the number of stages as long as you can keep feeding them.

But **autoregressive LLM decoding** is a **cyclic pipeline**, and this changes the story.

Even with **infinite incoming requests**, the amount of **in-flight work per decode step** on each pipeline stage (each individual worker) has a **hard upper bound**:

- To generate each new token, we must read that sequence’s **KV cache in VRAM**.
- We can only **evict** KV entries after many rounds of autoregressive generation.
- **VRAM is finite**, so the total number of KV tokens we can store is bounded.
- System-wide, the number of active sequences in flight at any step is bounded by the **smallest per-stage KV capacity** .

Call this bound “max in-flight sequences = 16” in a concrete example. That means:

- At most **16 sequences** can participate in decoding at once.
- The **minimum microbatch size** is 1, so the **maximum number stages with non-empty microbatches** is 16.
- If we have **more than 16 stages**, some stages must be idle—there just aren’t enough sequences to give each stage its “own” microbatch at any moment.
- Those idle periods are **bubbles**, which cut effective throughput roughly in proportion to the idle fraction

For instance, with 32 stages and only 16 sequences in flight, at best **16 out of 32 stages** can be doing useful work at a time, so the hardware utilization, and effective throughput are upper-bounded by: $\frac{16}{32} = 50\%$

The deeper the pipeline (larger S), the worse this gets: the number of microbatches needed in flight grows $\propto$ S, but the total in-flight sequences is capped by the KV budget.

So, in cyclic pipelines like LLM decoding:

> Deepening the pipeline is not free. With a fixed KV budget, there is a hard limit on how many microbatches (and thus stages) you can keep fully occupied. Beyond that point, more stages mainly introduce bubbles and **decrease effective throughput.**

Later, when we analyze pipeline parallelism for generation (even with KV swapping), we’ll see this constraint show up directly in how we choose the number of stages and microbatches.


---

# 3.  Making Pipeline Parallel More Efficient

We’ve established two facts:

1. **Decode is memory-bandwidth–bound**.
2. **Throughput is set by the slowest stage** in a pipeline once it’s warm.

## **3.1 Make the work non-blocking: More Micro Batches**

For a **single microbatch** on one device, the flow looks like:

1. **Compute - r**un the local layers on the incoming activations:
    
    $[B_{\text{micro}}, T, d_{\text{model}}] \rightarrow [B_{\text{micro}}, T, d_{\text{model}}]$.
    
2. **Upload -** Enqueue the output activations to be sent to next peer (**non-blocking** send).
    
3. **P2P (on the wire) -** the activations move over NVLink / PCIe / Ethernet / Internet to the next device.
    

On the **receiving** side, **download** is posted as a non-blocking receive and overlapped with decode of previous microbatches, so we do **not** include a separate $t_{\text{download}}$ in the critical path.

For throughput analysis, that leaves us with **three potentially blocking substeps per stage and per microbatch**:

- $t_{\text{decode}}$ – local forward pass (compute + HBM traffic),
- $t_{\text{upload}}$ – time to push activations onto the link,
- $t_{\text{p2p}}$ – pure network latency.

The effective per-stage time is therefore

$t_{\text{stage}} \approx \max\big( t_{\text{decode}},\ t_{\text{upload}},\ t_{\text{p2p}} \big)$, where $t_{\text{decode}} = \max(t_{\text{io}},\ t_{\text{compute}})$.

In steady state, the system throughput is simply microbatch size divided by the largest per-stage time across devices:

$\text{throughput} \;\approx\; \frac{B_{\text{micro}}}{\max_i t_{\text{stage}, i}} =

\frac{B_{\text{micro}}}{ \max_i \big( \max\big(t_{\text{decode},i},\ t_{\text{upload},i},\ t_{\text{p2p},i}\big) \big) }.$

To keep a pipeline with $N$ devices and **3 substeps** per device (decode → upload → p2p) fully busy, we need on the order of

$M \;\gtrsim\; 3 \times N$

microbatches in flight.

Intuitively: while one microbatch is decoding on a device, another can be uploading, and another can be in P2P transit, so we need roughly “`#substeps × #stages`” microbatches to fill the whole space.

---

## What `Throughput` Are We Comparing

Before we throw numbers around, we need to be precise about _what_ we’re comparing.

- **Metric: tokens per second.**
    
    In steady state, the pipeline emits $B_{\text{micro}}$ tokens per decoding step (one per sequence in the microbatch). If the slowest expanded stage (compute → upload → P2P) takes $t_{\text{max stage}}$ seconds per step, then:
    
    $\text{throughput} \;\approx\; \frac{B_{\text{micro}}}{t_{\text{max stage}}}, \quad t_{\text{max stage}} = \max_i t_{\text{stage}_i}$.
    
- **Workload regime: offline, heavy load.**
    
    We assume:
    
    - high concurrency and/or long context;
    - enough in-flight microbatches to keep all stages busy;
    - no idle gaps due to lack of requests.
- **Warm pipeline.**
    
    We ignore:
    
    - the initial warm-up (filling the pipeline),
    - the tail (draining the last microbatches).
    
    We only care about the **steady-state throughput** once the pipeline is full.
    
- **Same hardware, different network.**
    
    We compare two setups that use **identical GPUs and the same model**:
    
    - **Centralized**: all GPUs in one chassis with **fast links** (e.g., NVLink / NVSwitch).
    - **Decentralized**: GPUs spread across machines / locations with **slower links** (e.g., PCIe-only, Ethernet, or WAN).
    
    The only thing we change is the **P2P behavior** (latency and bandwidth).
    
    Our question is:
    
    > Given the same GPUs and model, can decentralized PP match the tokens/s of a centralized NVLink box, as long as we can hide P2P under decode?
    

That’s exactly the relationship we’ll formalize next.

---

## **Hiding P2P Latency - NVLink or WLAN yields SAME throughput**

Recall our expanded stage for one decoding microbatch:

- **decode** – local forward pass (compute + HBM traffic on the layers this device owns),
- **upload** – stream the resulting activations off the device and onto the network link.
- **P2P** – network latency from “last byte leaves sender” to “tensor is ready to use” on the next device.

$t_{\text{stage}} \approx \max\big( t_{\text{decode}},\ t_{\text{upload}},\ t_{\text{p2p}} \big), \quad t_{\text{decode}} = \max(t_{\text{io}}, t_{\text{compute}}).$

Our goal for high throughput is simple:

> **Keep every working devices busy →** make sure decode remains the bottleneck

Formally, on every pipeline stage, we need

$t_{\text{decode}} \;\ge\; t_{\text{upload}},\ t_{\text{p2p}}$.

If that holds on every device, the slowest stage in the _whole pipeline_ is still a **decode** stage, not the network, so centralized (NVLink) and decentralized (LAN/WLAN) setups have the **same tokens/s** on the same GPUs.

---

### Decode Time - Under Heavy Workload

From the GEMM analysis, decode is **memory-bound** on modern GPUs. We can summarize per-stage decode time for one microbatch as:

$t_{\text{decode}} \approx \frac{\text{Bytes}{\text{HBM per step}}}{\text{BW}{\text{HBM}}}$,

where

- $\text{Bytes}_{\text{HBM per step}}$ is the total HBM traffic for a microbatch running through all layers a single devices host in one decode step (activations + KV cache + weights touched),
- $\text{BW}_{\text{HBM}}$ is the device’s effective HBM / GDDR bandwidth.

We now build a conservative, microbatch-aware model of $\text{Bytes}_{\text{HBM per step}}$.

**Memory split (per GPU)**

We assume a typical inference layout:

- $r_{\text{weight}} \approx 50\%$ of VRAM holds **weights**,
- $r_{\text{kv}} \approx 40\%$ holds the **KV cache pool**,
- the remaining ~10% is “other stuff” (temporary activations, fragmentation, etc.).

So for a GPU with total memory $\text{VRAM}$:

- weights ≈ $0.5 \cdot \text{VRAM}$,
- KV pool ≈ $0.4 \cdot \text{VRAM}$.

**KV traffic and microbatch ratio**

KV capacity in _tokens_ for this stage is:

$N_{\text{tokens,pool}} \approx \frac{0.4 \cdot \text{VRAM}}{\text{Bytes}_{\text{KV per token}}}$,

where $\text{Bytes}_{\text{KV per token}} = 2 \times L \cdot (H_{kv} \cdot S_{kv} \cdot d_{head})$ accounts for K+V, local layers, and head dim.

At any time, these tokens are spread over many active sequences. Let:

- $N_{\text{seq,total}}$: sequences currently resident in this KV pool,
    
- $N_{\text{seq,micro}}$: sequences included in the current microbatch on this stage,
    
- **microbatch ratio**:
    
    $\rho_{\text{mb}} = \frac{N_{\text{seq,micro}}}{N_{\text{seq,total}}}$.
    

Recall that, to keep each stage busy, number of micro batches should at least be `n_substeps * n_pp_stages` ; given we have 3 substeps, and suppose our pipeline consists of 3 working nodes, then $\rho_{\text{mb}} = \frac{1}{10}$

Intuitively: each microbatch “owns” roughly one slice out of the ~9–10 slices that cover all active sequences whose KV live on this stage.

Let’s assume sequences are roughly similar in length, $\rho_{\text{mb}}$ is also the approximate **fraction of KV pool we touch** in one decode step:

$\text{KV traffic per step} \;\approx\; \rho_{\text{mb}} \cdot (0.4 \cdot \text{VRAM}) = 0.04 \cdot \text{VRAM}$

**Weight traffic: we always use all weights**

On the weight side, for a given stage we run **all its local layers** on the microbatch every decode step:

- Q/K/V projections,
- attention output projections,
- MLP up/down projections.

Even though weights are reused across tokens in the batch via tiling, from the HBM perspective we effectively stream the **entire weight footprint** of that stage once per step in the worst case, so we have weight traffic per step around $0.5 \cdot \text{VRAM}$

**Putting Weights and KV Together**

$t_{\text{decode}} \approx \frac{r_{\text{weight}} + r_{\text{kv}} \cdot \rho_{\text{mb}} \cdot \text{VRAM}}{\text{BW}_{\text{HBM}}} = \frac{0.54 \cdot \text{VRAM}}{\text{BW}_{\text{HBM}}}.$

---

### **A Tension: Batch Size, Stage Depth, and How Much Memory We Actually Touch**

> We can’t simultaneously maximize batch size, minimize stage depth, and perfectly fill the KV pool at every step. Pipeline parallelism buys us model-capacity and throughput, but it also fragments the workload across microbatches and stages. Our “decode window” estimates are thus **upper bounds** based on a reasonably busy regime, not guarantees that we literally stream all KV and all weights each step.

So far, we’ve argued:

- Larger **batch size** improves hardware utilization:
    
    we reuse weight tiles across more tokens, we keep the SMs busier, and decode becomes very cleanly memory-bound.
    
- In pipeline parallel, to keep all stages busy, we need enough **microbatches in flight**:
    
    $M \gtrsim n_{\text{substeps}} \times n_{\text{pp\_stages}}$
    

But these two facts pull in opposite directions once we also account for **memory layout** (weights vs KV).

### **1. Most of the VRAM is weights, but microbatching limits reuse**

In our model:

- ≈50% of VRAM is **weights** (per PP stage).
- ≈40% is **KV cache pool**.

On a **single device** with no pipeline:

- You can choose a large batch size,
- Decode each step can stream “all weights once, reused across a big batch,”
- And KV reads can cover a large fraction of the KV pool (many sequences, long contexts),
- So a **big chunk of VRAM gets touched per step**, and utilization looks great.

Once you introduce **pipeline parallel** and a fixed total batch, things change:

- To fill a pipeline of S stages with $n_{\text{substeps}} = 3$, you need $M \sim 3S$ microbatches.
    
- For fixed global batch B, the **microbatch size** is:
    
    $B_{\text{micro}} = \frac{B}{M}$.
    
- As you increase S (more PP stages), M grows and $B_{\text{micro}}$ **shrinks**.
    

That means:

- Each stage **still needs all of its weights**, so you _always_ pay to stream that stage’s full weight block each step.
    
- But each microbatch now covers only a **small slice of the sequences** whose KV live on that stage:
    
    $\rho_{\text{mb}} \approx \frac{1}{M} \approx \frac{1}{3 S}$.
    
- So per step, you only touch a $\rho_{\text{mb}}$ **fraction of the KV pool**, and that fraction shrinks as you deepen the pipeline.
    

Net effect: you are **paying full price for weights** but **underutilizing** the KV pool and batch-level reuse as PP gets deeper. Decode is still memory-bound (weights alone can do that), but the “we stream most of VRAM per step” picture gets weaker.

### **2. If you reduce stages, you lose overlap and go back to compute+upload**

The obvious idea is: “OK, just use fewer PP stages so microbatches can be larger.”

But fewer stages mean:

- Each device holds **more layers**, i.e., more weights.
- If you keep the same total batch and kv memory, you may now **run out of VRAM** unless you shrink the KV pool (fewer sequences, shorter contexts, or both).

And if you also treat **upload as blocking** (no real overlap between decode and upload):

- A stage timeline becomes:
    
    $t_{\text{stage}} \approx t_{\text{decode}} + t_{\text{upload}}$,
    
    not $\max(\text{decode}, \text{upload}, \text{p2p})$.
    
- Overall throughput is then limited by **compute+upload in series**,
    
    not by the nicer “max(decode, comm)” ideal we were shooting for.
    

So: fewer stages reduce microbatch fragmentation, but if you don’t aggressively overlap comm, you still leave utilization on the floor—now because communication is serialized with compute, rather than because KV reuse is fragmented.

### **3. If you shrink weights per device, you add stages and fragment KV even more**

The opposite extreme is: “Let’s shrink the per-device weight footprint so we can fit a much bigger global batch / KV pool.”

To do that, you:

- Slice the model into **more PP stages** (each stage holds fewer layers / weights).
    
- That **increases** S and hence the required number of microbatches M \sim 3S.
    
- Which **shrinks** each microbatch’s share of the KV pool even further:
    
    $\rho_{\text{mb}} \approx \frac{1}{3S}$,
    
    so each step touches **less** of the KV pool, not more.
    

So you gain KV capacity in theory, but then carve it up into more stages and more microbatches, which limits per-step KV reuse and keeps you from “reading the whole pool” as aggressively as you might want.

---

### **The Takeaway**

All of this is the same tension, seen from different angles:

- **Big batches** and **few stages** are great for weight reuse and KV coverage, but strain memory and limit how much you can split the model across devices.
- **More stages** reduce weight memory per device, but force more microbatches in flight, which:
    - shrinks $B_{\text{micro}}$,
    - shrinks $\rho_{\text{mb}}$,
    - and makes it harder to justify “we touch a huge fraction of the KV pool each step.”

---

### Upload Time - With 100Mbps Assumption

- $t_{\text{upload}} = t_{\text{download}} \approx \frac{\text{Bytes}_{\text{act}}}{\text{BW}_{\text{p2p\ up/download}}}$ where
    - $\text{Bytes}_{\text{act}}\approx 2 \cdot d_{\text{model}} \cdot T_{\text{in-flight}}$ is the size of the activation tensor being transmitted, for bf16.
    - $\text{BW}_{\text{p2p}}$ be the effective p2p upload/download bandwidth.

Taking in our 32B model, we have

$\text{Bytes}_{\text{act}} \approx 2 \cdot 5120 \cdot 32 \approx 0.32\ \text{MB}.$ With 100Mb/s bandwidth, we have

$t_{\text{upload}} = \frac{0.32\ \text{MB}}{12.5\ \text{MBps}} = 28\ \text{ms}$

---

# NVLink - TP v.s. PP

(Do we still need this section?)

Latency matters.

Also, we show that, with high p2p latency, PP **can’t** actually makes working GPUs running with high utilization.

---

# What about MACs

So far we’ve focused on **GPUs** (4090/5090/A100), where:

- HBM bandwidth is huge,
- decode per microbatch per stage is on the order of **a few–10 ms**,
- and network (upload + P2P) needs to be carefully sized to fit under that window.

On **Mac / edge devices**, the picture becomes strictly _easier_ for hiding P2P:

- The **same model** running on a Mac (M-series, unified memory) has **much lower FLOPs and memory bandwidth** than a 4090/A100.
- For the same microbatch and layer stack, **decode time per step is significantly longer** (often 5–10× or more), simply because the hardware is slower.

### **1. On Mac, P2P is dwarfed by decode ⇒ 2 substeps are enough**

For Mac:

$t_{\text{decode,Mac}} \gg t_{\text{upload,Mac}},\ t_{\text{p2p,Mac}}$.

In that regime, we can safely **collapse back to a 2-substep model**:

$t_{\text{stage,Mac}} \approx \max\big(t_{\text{decode,Mac}},\ t_{\text{comm,Mac}}\big) \approx t_{\text{decode,Mac}}$,

### **2. Even with smaller effective microbatches, we still hide latency**

As before, to keep a pipeline with S stages busy, we need on the order of:

$M \gtrsim n_{\text{substeps}} \times S$

microbatches in flight. On GPUs we used **3 substeps** (decode → upload → p2p) and saw that this constrains microbatch size and the KV fraction we touch per step.

On Mac stages, we can:

- Treat the stage as effectively **2 substeps**: decode → “p2p+upload”.
- Or more informally: we only really care about **decode** as the critical substep; communication is fully hidden inside its much longer window.

This means:

- Even if the **effective microbatch ratio** on Mac is small (because the overall pipeline is deep),
    
- The **decode window is so long** that:
    
    $t_{\text{upload,Mac}},\ t_{\text{p2p,Mac}} \ll t_{\text{decode,Mac}}$,
    
    and both are easily hidden.
    
- You don’t need a huge microbatch per Mac stage to amortize its P2P latency; the slower hardware _itself_ gives you a generous margin.
    

---

# **Micro-Batching + PP, High P2P v.s. Low P2P on _MAC_**

For datacenter GPUs, careful bandwidth/latency analysis is needed to hide P2P under decode. For Macs and similar edge hardware, **decode is so slow** relative to LAN/WLAN networking that P2P is trivially hidden—even with smaller effective microbatches—so decentralized PP over Macs is _even more_ likely to converge to the same throughput as a centralized setup on the same devices.

We may argue that, even for MAC, we favor large batch size per decode step, to reuse the weights loaded.

Suppose we store 50% VRAM with weights, and our inflight batch takes all the KV cache pool;

1. if we don't do micro batch, then we use a single decode step with max batch size to finish compute, this way, we takes around `memory / io_bandwidth`;
2. if, instead we use microbatch size of 2, then we are taking (`1/2 [weight] + 1/4 [half of inflight requests]) memory / io * 2 => 1.5memory / io`, which is **50%** waste. So, in general, micro-batching for bandwidth-bound scenario is extremely limited.

Now, let’s further suppose we have max batch size 16 and pp stage of 2.

PP without micro batching takes 8 * 3 = 24 blocks to complete 3 rounds of decoding; each round produces 16 tokens.

Now with micro batching, we split max batch size of 16 to 2 micro batch sizes of 8. Each micro batch (blue, then purple), takes 3 blocks. So a single round of 16 sequences take 9 blocks (instead of 8). However, when the pipeline is warm, we effectively need only **6** blocks instead of **8** to finish a round of 16 sequences.

![image.png](attachment:3fb4b536-247f-4639-885d-5369e6868682:image.png)

## If P2P latency is relatively large, compared to per decode:

We didn’t gain much when we have high P2P latency even if we do micro batching; if P2P is relatively high compared to per decode round latency.

- When p2p is relatively small (1/4 of the max batch decode time)

![image.png](attachment:581661d9-7475-40d4-a62e-d21c1c18da2b:image.png)

- When p2p is roughly the same as the time taken per decode round:

![image.png](attachment:50d42ed9-a29f-4a12-b40c-fc5d3e87af5f:image.png)

- Extreme case: p2p takes 2x per decode round. Now, even with MB, we still fall short in terms of working node utilization.

![image.png](attachment:37c8b669-c19f-41ec-99ae-b012653b9f03:image.png)

We observe that, no matter how large is P2P latency, if we use 2-stage approach, we’ll end up with `2 * micro_batch_decode_latency + p2p_latency`

# Hardware Spec

```jsx
4090: 
RAM: 24G, Bandwidth: 1.01T/s, TFLOPS (fp16): 165.2, Price: $1600

5090:
RAM: 32G, Bandwidth: 1.79T/s, TFLOPS (fp16): 209.5, Price: $2000

A100-40G:
RAM: 40G, Bandwidth: 1.55T/s. TFLOPs (TensorCore fp16,bf16): 312, 
Price: $8k - $10k, DGX A100: $200K
(NVLink 600G/s)

H100-SXM:
RAM: 80G, Bandwidth: 3.35T/s, TFLOPs 1979
Price: $25K - $40K; DGX H100: $300-400K
(NVLink 900G/s, PCIE5, 128G/s)

MAC - M4 ???
RAM: 64G, Bandwidth: 274G/s (unified memory to ANE sram?), 
TFLOPs: 38 (Apple NeuronEngine, similar to Tensor Cores)
```