# 0. Overview

In this tech blog, we systematically analyze the feasibility of hosting Large Language Models (LLMs) on edge devices using **Pipeline Parallelism (PP)**. We specifically target scenarios where **Tensor Parallelism (TP)** is unfeasible due to slow interconnects (e.g., PCIe, Ethernet, or WiFi).

We explore two specific optimization strategies to maximize throughput:

1. **Global Micro-Batching (Inter-Device):** Splitting the workload between different GPUs to minimize idle time ("bubbles").
    
2. **Local Micro-Batching (Intra-Device):** Splitting the workload on a _single_ GPU to overlap Compute, Upload, and P2P transfer, aiming to hide network latency.
    

**The Executive Summary:** We demonstrate that while **Global Micro-Batching** is essential for multi-device utilization, **Local Micro-Batching** is largely a "performance trap" on memory-bound consumer hardware.

Due to the **"Latency Floor,"** shrinking the batch size on a single device does not reduce decode latency. Consequently, aggressive local pipelining forces the GPU to reload massive weight matrices repeatedly without speeding up token generation—effectively "spinning" the GPU to hide network latency, but starving it of memory bandwidth in the process.

Therefore, for consumer hardware, the optimal strategy is **Global Micro-Batching only**. We must accept the P2P latency cost rather than sacrificing compute efficiency to hide it.

**The Roadmap:** We will start by establishing the Memory Bandwidth Bottleneck (Section 1), move to the necessity of Global Pipelining (Section 2), and finally prove why Local Pipelining fails to solve the P2P bottleneck on edge devices (Section 3).



### **Final conclusions (high-level):**

- **TP is not feasible** on edge hardware (every layer requires all-reduce; PCIe/LAN latency kills performance).
    
- **PP is the only viable strategy** for model parallelism on consumer GPUs, just as in datacenters — TP is used only when NVLink/NVSwitch is available.
    
- **Global micro-batching** (inter-device pipelined execution) **works** and improves throughput.
    
- **Local micro-batching** (intra-device 3-stage overlap) **does not work**, because:
    
    - batch fragmentation must increase to enable overlap,
        
    - decode latency does **not** shrink with batch size (memory-bound regime),
        
    - weight streaming dominates, creating a **latency floor**, and
        
    - the GPU ends up “busy but useless,” repeatedly streaming the same weights for many tiny microbatches.
        
    

  

Therefore:

  

> **On consumer hardware, the only practical optimization is global pipeline parallelism with a moderate number of micro-batches. Local pipelining to hide P2P latency is not feasible and usually makes throughput worse.**

  

This frames the rest of the document: everything after Section 1 explains _why_ decode is memory-bound, and everything after Section 2 shows _why_ local pipelining fails despite appearing promising in theory.

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

# 2. Pipeline Parallel and Micro Batching

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

To fix this, we need **Micro-batching**—slicing the batch into smaller chunks to keep multiple devices active simultaneously. 


---

## **2.4 Micro Batching**

In the "naive" pipeline approach described earlier, we treated the entire batch as a single unit. This leads to a stop-and-wait behavior: while GPU 1 is processing the batch, GPU 2 is completely idle, waiting for data.

To fix this, we split the global batch $B$ into smaller **micro-batches** that flow through the system independently.

Consider a 2-GPU pipeline with a global batch size of 16.

Instead of sending all 16 sequences at once, we split them into 2 micro-batches of 8 sequences each.

**The Flow:**

1. **Step 1:** GPU 0 processes Micro-Batch 1 (MB1). GPU 1 is idle (unavoidable warm-up).
    
2. **Step 2:** GPU 0 passes MB1 to GPU 1.
    
    - _Crucially_, GPU 0 does not wait. It immediately begins processing **Micro-Batch 2 (MB2)**.
        
    - **Result:** Now **both** GPUs are working simultaneously. GPU 1 works on the first chunk (MB1), while GPU 0 prepares the second chunk (MB2).
        

By slicing the work, we minimize the time where workers are waiting on peers. This "pipelining" effect is the primary driver of efficiency in multi-device inference.


## **2.5 Performance Analysis**

Let's now construct a concrete case, and we will be using this example through out our note.

As established in previous section, decoding is memory-bandwidth bound, so we are using only IO time here for our analysis.

**Assumptions:**

- **Hardware:** 2 GPU Pipeline ($N=2$).
    
- **Memory Split:** 50% Weights ($W$), 50% KV Cache ($KV$).
    
- **Workload:** We want to process a Max In-Flight Batch Size of **16**.
    
- **Cost Unit:** Let **4 Blocks** represent the time to stream the entire VRAM (Weights + Full KV Pool) once.


![[PP_with_MB.png]]

### **Scenario A: Naive Pipeline (No Micro-Batching)**

We process the full batch ($B=16$) as a single monolithic step.

- **Data Touched:** 100% Weights + 100% KV Pool.
    
- **Cost per Step:** 4 Blocks (Full IO).
    

**The Flow:**

1. **GPU 0** processes the batch: **4 Blocks**.
    
2. **P2P Transfer.**
    
3. **GPU 1** processes the batch: **4 Blocks**.
    

Because GPU 1 cannot start until GPU 0 finishes the entire batch, the time to generate tokens for these 16 sequences is serialized.

$$\text{Total Cost} = 4_{\text{gpu0}} + 4_{\text{gpu1}} = \mathbf{8 \text{ Blocks}} + \text{P2P}$$
### **Scenario B: Micro-Batching (MB=2)**

We split the batch of 16 into two micro-batches of 8 ($MB_1, MB_2$).

- **Data Touched per MB:** 100% Weights ($0.5 \text{ VRAM}$) + 50% KV Pool ($0.25 \text{ VRAM}$).
    
- **Total Data:** 75% of VRAM.
    
- **Cost per Step:** $0.75 \times 4 \text{ Blocks} = \mathbf{3 \text{ Blocks}}$.
    

The Flow (Visualized):

In the diagram below, Blue represents MB1 and Purple represents MB2.

1. The Cold Start (Latency Increase):

If we look at the time to finish the very first round from a cold start:

- $T=0$: GPU 0 starts MB1 (Blue). [Cost: 3]
    
- $T=3$: GPU 0 finishes MB1.
    
    - GPU 0 starts MB2 (Purple). [Cost: 3]
        
    - GPU 1 starts MB1 (Blue). [Cost: 3]
        
- $T=6$: GPU 1 finishes MB1 (First 8 tokens out). GPU 1 starts MB2 (Purple). [Cost: 3]
    
- $T=9$: GPU 1 finishes MB2 (All 16 tokens out).
    

Total Cold Start Cost: $3 + 3 + 3 = \mathbf{9 \text{ Blocks}}$.

Observation: $9 > 8$. For a single isolated run, micro-batching is indeed slower due to the overhead of loading weights twice ($2 \times 0.5W = 1.0W$) versus once ($0.5W$).

2. The Steady State (Throughput Gain):

However, in a continuous generation loop, the pipeline is full. We don't wait for cold starts.

- GPU 1 is the bottleneck.
    
- It processes MB1 (3 Blocks) then immediately MB2 (3 Blocks).
    
- **Total Throughput Cost:** $3 + 3 = \mathbf{6 \text{ Blocks}}$.
    

**Comparison:**

- **Naive:** 8 Blocks per round.
    
- **Micro-Batching:** 6 Blocks per round.
    
- **Result:** Micro-batching provides a **33% speedup** (6 vs 8) in steady state.

### **2.5.1 Sensitivity Analysis: What if Weights are Huge?**

One might argue: "Micro-batching loads weights multiple times. If weights dominate VRAM, won't this kill performance?"

Let's check the extreme case: **80% Weights, 20% KV.**

- **Naive Cost (100% RAM):** 1.0 unit.
    
- **MB Cost (Single Step):**
    
    - Weights: 0.8 (Must load all).
        
    - KV: $0.1$ (Half of the 20% pool).
        
    - Total: $0.9$ units per MB.
        
- **MB Steady State (2 steps):** $0.9 \times 2 = \mathbf{1.8 \text{ units}}$.

Even with massive weights, $1.8 < 2.0$. Unless the overhead is super-linear or synchronization costs are extreme, micro-batching should mathematically always yield higher throughput.

### **2.5.2 P2P**

We assume:

- **$P$:** The time to transfer the full batch over the network (Red Block).
    
- **Blocking:** Compute and P2P do _not_ overlap on a single device.
    


![[Untitled-{{date}}-{{time}}.png]]

In Steady State, Naive Pipeline takes $8 + 2 \times P$ blocks; where as Micro Batching takes $2 \times (3 + P)$ blocks.


![[Untitled-{{date}}-{{time}}-1.png]]

![[Untitled-{{date}}-{{time}}-2.png]]

From the table below, we show that as P2P increases, the throughput improvement of using micro-batches decreases.

| P   | MB  | Naive | Improvement |
| --- | --- | ----- | ----------- |
| 1   | 8   | 10    | 25%         |
| 4   | 14  | 16    | 14%         |
| 8   | 22  | 24    | 9%          |


---

# **3. Hiding P2P Latency -  Feasibility of Intra-Device Pieplining**

## **3.1 Hiding Latency: Local Pipelining**

Micro-batching solves the "idle worker" problem, but on consumer hardware, we face a second bottleneck: **Slow Interconnects (P2P)**.

If a device computes a micro-batch and then stops to transfer data over a slow PCIe/LAN connection, we are still wasting time. We want to hide this communication cost.

We may achieve this by pipelining **within a single device**. i.e. other than **inter-devices** micro-batch, we introduce **Intra-Device** micro-bathching.

By using asynchronous streams, a single device can overlap different types of work. While the "Compute Engine" works on the _current_ micro-batch, the "Copy Engine" (DMA) can transmit the _previous_ micro-batch.

To analyze this, we model the execution of a single stage as **three distinct, non-blocking substeps** per micro-batch:

1. Decode ($t_{\text{decode}}$): The local forward pass (Compute & HBM access).
    
    $$[B_{\text{micro}}, 1, d_{\text{model}}] \rightarrow [B_{\text{micro}}, 1, d_{\text{model}}]$$
    
2. **Upload ($t_{\text{upload}}$):** Moving the activation tensor from GPU memory to the network buffer.
    
3. **P2P ($t_{\text{p2p}}$):** The "on-the-wire" transmission to the next device.

---

## **3.2 Hiding P2P Latency Completely**
With this 3-substep overlap, 
- while computing **MB $k$**,
    
- we are uploading **MB $k-1$**,
    
- and the network is transmitting **MB $k-2$**.
the effective time a stage takes to process a micro-batch is determined by the **slowest** of the three steps, not the sum:

$$t_{\text{stage}} \approx \max\big( t_{\text{decode}},\ t_{\text{upload}},\ t_{\text{p2p}} \big)$$

This leads to a powerful theoretical conclusion:

> **If we can keep the local decode step slower than the network transmission ($t_{\text{decode}} \ge t_{\text{p2p}}$), the slow network effectively disappears.**

In this ideal scenario, a decentralized cluster connected via slow Ethernet/WLAN would achieve the **exact same throughput** (tokens/sec) as a centralized cluster connected via NVLink, provided they use the same GPUs. The network latency is fully "hidden" behind the compute wall.

We could also do a 2-stage local pipelining, instead of 3, where we view `upload` and `p2p` as a whole.

 ---


## 3.3 Feasibility Analysis: Efficiency Trap

So, why isn't everyone running massive LLMs on consumer clusters with perfect efficiency? 

Essentially, we've established that
- Larger **batch size** improves hardware utilization:
    
    we reuse weight tiles across more tokens, we keep the SMs busier, and decode becomes very cleanly memory-bound.
    
- In pipeline parallel, to keep all stages busy, we need enough **microbatches in flight**:
    
    $M \geq n_{\text{local\_stages}} \times n_{\text{global\_stages}}$
    

But these two facts pull in opposite directions once we also account 


To hide network latency, we attempt **local micro-batching**, aiming to transform the global step time
$$
T_{\text{global}} = T_{\text{decode}} + T_{\text{p2p}}
$$
into the overlapped form
$$
T_{\text{local}} = \max\big(T_{\text{decode}}’,\, T_{\text{p2p}}\big), \qquad T_{\text{global}} = n_{\text{local\_stages}} \cdot T_{\text{local}}.
$$
The strategy requires shrinking micro-batch size so that
$$
T_{\text{decode}}’ > T_{\text{p2p}},
$$

thus allowing communication to be “hidden” under a slower decode phase.

  
However, on memory-bound hardware:
$$
T_{\text{decode}}’ \approx  T_{\text{decode}} \quad \forall\, B_{\text{micro}},
$$

because weight streaming dominates and does not decrease with smaller batch size.

The GPU ends up repeatedly reloading the same weights for many tiny micro-batches—keeping busy, but not producing tokens faster.

In our block-analysis in section 2.5: if $T_{\text{decode}} = 4$ blocks and $T_{\text{p2p}} = 4$ blocks, then:

- Before: 4 + 4 = 8 blocks per step
    
- After “overlap”: $\max(4,4) \times 2$ = 8 blocks per step — **no improvement**
    

We are caught in a paradox:
To hide the network, we must shrink micro-batches; shrinking micro-batches starves the GPU and eliminates any benefit.

### **3.3.1 The Cost of Overlap: Logical Depth**

To keep the 3 local pipeline stages (Decode, Upload, P2P) fully busy simultaneously, we need distinct data for each step, we need at least 3 micro-batches "in flight" locally.

If we have a pipeline of $N=2$ GPUs, and we want full 3-step overlap on both, we effectively have a Logical Pipeline Depth of $3 \times N = 6$.

To avoid bubbles, we must split our Global Batch $B$ into at least 6 tiny chunks:

$$B_{\text{micro}} \approx \frac{B}{6}$$

Each GPU is now processing only $1/6$ of the original batch size at any given moment.

### **3.3.2  Weights Thrashing**

This fragmentation is fatal for memory-bandwidth-bound tasks like decoding because of the **Static Weight Penalty**.

Recall our memory breakdown:

- **Weights:** $\approx 50\%$ of VRAM. Static. Must be loaded **every single decode step**.
    
- **KV Cache:** $\approx 40\%$ of VRAM. Dynamic. Traffic scales with batch size.
    

**The "Death by Fragmentation" Loop:**

1. **The Goal:** Hide slow P2P ($t_{\text{p2p}}$ is high).
    
2. **The Action:** We increase the number of micro-batches ($M$) to create overlap.
    
3. **The Consequence:** The micro-batch size ($B_{\text{micro}}$) shrinks drastically.
    
4. **The Efficiency Crash:**
    
    - We still pay the "tax" of loading **100% of the weights** (12GB+) for every tiny micro-batch.
        
    - But we only generate a tiny sliver of tokens (useful work).
        
    - **Result:** The GPU spends 90% of its memory bandwidth reloading weights and only 10% reading/writing user data.


In real deployment, if the **Weights** dominate the transfer, the time to decode Batch=1 is almost identical to the time to decode Batch=4.

$$T_{\text{decode}}(B=1) \approx T_{\text{decode}}(B=4)$$

This creates a **Latency Floor**. Until you reach a large enough batch size to saturate the compute units or make KV traffic significant, **increasing batch size comes at "zero cost" in latency.** Conversely, **reducing batch size saves zero time.**


### **3.3.3 What about Fewer Stages?**
A natural counter-proposal is to reduce the pipeline depth—either by using fewer physical devices (Global) or fewer overlapping substeps (Local)—to allow for larger, more efficient micro-batches. However, both approaches introduce new bottlenecks.

#### **Fewer Global Stages (Inter-Device)**

Reducing the number of GPUs (e.g., splitting a model across 2 GPUs instead of 4) reduces micro-batch fragmentation, but introduces severe memory constraints:

- **Increased Weight Footprint:** Each device must hold a larger fraction of the model parameters.
    
- **KV Starvation:** With weights consuming more VRAM, the remaining budget for the KV cache shrinks. This creates a hard ceiling on maximum sequence length and concurrency, potentially forcing an Out-Of-Memory (OOM) error even at moderate workloads.
    
- **Reduced Effective Batch Size:** To fit the model, we may be forced to lower the batch size, which ironically leads us back to the "Latency Floor" inefficiency we tried to avoid.
    

#### **Fewer Local Stages (Intra-Device)**

Alternatively, we could simplify the intra-device pipeline by collapsing the 3-substep model (Decode $\to$ Upload $\to$ P2P) into 2 steps.

- **Loss of Latency Hiding:** This reintroduces serialization. For example, if we treat "Upload + P2P" as a single atomic block, the high-speed Copy Engine (Upload) is effectively blocked by the slow Network Interface (P2P). We lose the ability to prepare the next packet while the current one is on the wire, re-exposing the P2P latency we aimed to hide.
- **Upload Time - With 100Mbps Assumption**
	
	- $t_{\text{upload}} = t_{\text{download}} \approx \frac{\text{Bytes}_{\text{act}}}{\text{BW}_{\text{p2p\ up/download}}}$ where
	    - $\text{Bytes}_{\text{act}}\approx 2 \cdot d_{\text{model}} \cdot T_{\text{in-flight}}$ is the size of the activation tensor being transmitted, for bf16.
	    - $\text{BW}_{\text{p2p}}$ be the effective p2p upload/download bandwidth.
	Taking in our 32B model, we have
	
	$\text{Bytes}_{\text{act}} \approx 2 \cdot 5120 \cdot 32 \approx 0.32\ \text{MB}.$ With 100Mb/s bandwidth, we have
	
	$t_{\text{upload}} = \frac{0.32\ \text{MB}}{12.5\ \text{MBps}} = 28\ \text{ms}$

---


# 4. Conclusion: The Edge Inference Strategy

We started this analysis to see if we could make a cluster of consumer GPUs (connected via LAN/WiFi) perform as well as a datacenter cluster (connected via NVLink) by using aggressive pipelining to hide latency.

Our mathematical and cost-model analysis yields the following verdict:

### **1. Tensor Parallelism is Off the Table**

On slow interconnects, the O(L) synchronization overhead of TP is prohibitive. Pipeline Parallelism is the necessary fallback, reducing communication to O(1) per micro-batch.

### **2. Global Micro-Batching is Mandatory**

To prevent GPUs from sitting idle while peers work, we must split the global batch into at least M≈Ngpus​ micro-batches. This allows GPU 0 to process Batch k+1 while GPU 1 processes Batch k.

### **3. Local Micro-Batching is a Trap**

Attemping to hide the specific P2P latency (tp2p​) by further fragmenting the batch on a single device (Intra-Device Pipelining) fails due to the **Latency Floor**.

- **The Trap:** Decoding a batch of 1 takes the same time as a batch of 4 (dominated by loading weights).
    
- **The Consequence:** Splitting the batch just forces the GPU to load weights more often. You hide the network latency, but you inflate the compute time so much that the total throughput drops.
    


So for edge devices hosting LLMs:

1. **Maximize Batch Size:** Fill the "Latency Floor." Process as many tokens as possible per weight load.
    
2. **Accept the P2P Penalty:** Do not use intra-device overlapping if it requires reducing the batch size below the saturation point. It is better to have a fast compute step followed by a slow network step than to have a disastrously slow "overlapped" step.
    
3. **Optimize the Link:** Since we cannot hide the latency via compute, the only physical optimization remaining is to improve the link speed (e.g., upgrade from WiFi to Ethernet, or optimize compression of the activation tensors).