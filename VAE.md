---
tags:
  - Research
themes: Image Generation, Multi-Modal, Machine Learning
created: 2025-12-14
---
# Autoencoders

## Data distribution: the (approximate) manifold hypothesis
**Hypothesis**: high-dimensional observations $x \in \mathbb R^D$ concentrate near a low-dimensional, curved set $\mathcal M \subset \mathbb R^D$ with $\dim(\mathcal M) \ll D$.

- **Examples**:
	- Images: lighting, pose, identity, texture
	- Speech: phonemes, pitch, speaker identity
	- Text embeddings: semantics, syntax, style
- **Consequence**: random points in $\mathbb R^D$ are typically meaningless; only points near $\mathcal M$ look like data.
- **Generation viewpoint**: to generate realistic $x$, we need a strategy to sample points that land on/near $\mathcal M$.

## Local coordinates + projection back to $\mathcal M$
An autoencoder captures the manifold because **reconstruction pressure** forces the encoder to learn a low-dimensional coordinate system while collapsing off-manifold directions.

- ![[VAE.png]]
- **Encoder (compression)**: forces $x$ through a bottleneck $z$, discarding information not needed for reconstruction.
- **Decoder (reconstruction)**: maps latent coordinates back to the ambient space, ideally landing near the data distribution.

Formally, an autoencoder solves:

$$
\min_{\theta,\phi}\ \mathbb E_{x \sim p_{\text{data}}}\left[\lVert x - D_\phi(E_\theta(x))\rVert^2\right]
$$

where
- Encoder: $E_\theta: \mathbb R^D \to \mathbb R^d$
- Decoder: $D_\phi: \mathbb R^d \to \mathbb R^D$
- Bottleneck: $d < D$

### Local geometry: tangent vs normal directions 
A smooth manifold is globally curved but **locally linear**: around a point $x$, $\mathcal M$ is well-approximated by its **tangent space** $T_x\mathcal M$, a linear hyperplane. (Flat Earth Analogy)

Reconstruction loss induces two complementary behaviors:

- **Sensitivity along tangent directions** (meaningful variation):
	- Let $x' = x + \epsilon v$ with $v \in T_x\mathcal M$.
	- Then $x'$ is still (approximately) a valid data point, so reconstruction should stay low for both $x$ and $x'$.
	- Therefore the encoder must **distinguish** $x$ and $x'$: the latent representation should change meaningfully along tangent directions.

- **Insensitivity along normal directions** (implicit *denoising*):
	- Let $x' = x + \epsilon n$ with $n \perp T_x\mathcal M$.
	- Then $x'$ is off-manifold (no training data there).
	- Minimizing expected reconstruction encourages mapping $x'$ back toward (the neighborhood of) $\mathcal M$, behaving like a **projector**, even without explicit noise.

### Jacobian viewpoint (more formal intuition)
- **Encoder Jacobian aligns with the tangent space**:
	- *Row view*: rows of $J_{E_\theta}(x)$ are gradients of latent coordinates; they point in directions where $z$ changes, which should primarily be directions along $\mathcal M$.
- **Decoder Jacobian spans the local tangent plane at $\hat x$**:
	- With $\hat x = D_\phi(z)$, columns of $J_{D_\phi}(z)$ are $\frac{\partial \hat x}{\partial z_i}$, giving a basis for the local tangent directions of the decoded manifold.

## Noise & Interpolation in practice
- **Latent interpolation looks semantic**:
	- Let $z = E_\theta(x)$. For a small $\delta$, define $z' = z + \delta$, and decode $x' = D_\phi(z')$.
	- Since $D_\phi$ is trained only on (approximately) on-manifold codes, $x'$ tends to stay near $\mathcal M$, yielding realistic variations (pose, lighting, expression, …).
- **Off-manifold noise is collapsed**:
	- For $x' = x + \epsilon n$ (normal perturbation), typically $E_\theta(x') \approx E_\theta(x)$, so $D_\phi(E_\theta(x')) \approx D_\phi(E_\theta(x)) \approx x$.

## What Autoencoders is (and not) capable of
### Capability
* A manifold *parameterization* (local chart)
	* To parameterize a manifold $\mathcal M \subset \mathbb R^D$ means assigning each data point $x \in \mathcal M$ a coordinate $z \in \mathbb R^d$ such that nearby points on $\mathcal M$ have nearby coordinates.
	* An AE learns two maps:
		* Encoder: $E_\theta:\mathbb R^D \to \mathbb R^d$
		* Decoder: $D_\phi:\mathbb R^d \to \mathbb R^D$
		* Reconstruction constraint (on data): $D_\phi(E_\theta(x)) \approx x$ for $x \sim p_{\text{data}}$ (i.e. near $\mathcal M$).
	* Like charts in differential geometry, this is generally **local**—not guaranteed to be globally **invertible**.
* A projection operator onto the data manifold: $P(x) := D_\phi(E_\theta(x))$
	* For $x \in \mathcal M$: $P(x) \approx x$
	* For $x \notin \mathcal M$: $P(x)$ tends to map back toward (a nearby point on) $\mathcal M$
	* Geometrically: the encoder suppresses off-manifold components; the decoder reconstructs an approximately closest on-manifold point.
* Meaningful latent directions *locally*
	* A latent direction is a direction $v \in \mathbb R^d$ such that $D_\phi(z + \epsilon v)$ produces a meaningful change in data space (e.g. rotate a face, change lighting, modify pitch).
	* Around $z_0 = E_\theta(x_0)$, the decoder is approximately linear:

$$
D_\phi(z_0 + \delta) \approx D_\phi(z_0) + J_{D_\phi}(z_0)\,\delta,
$$

	so small moves can correspond to semantic changes, while large moves can leave the region the decoder was trained on.
	* Practically, AE latent spaces can contain *holes*, *folds / self-intersections*, and *disconnected regions*—so a “direction” meaningful near one sample may not generalize globally.

### Incapability
* A *probability distribution* over the manifold (and thus principled sampling)
	* We want to switch from a **point mapping** ($x \mapsto z$) to a **distribution** we can sample from.
	* In a vanilla AE, $z = E_\theta(x)$ and $\hat x = D_\phi(z)$ are deterministic, so there is no explicit density over $z$ (or $x$).
	* Implicitly we only have an empirical set of latent codes $\{z_i = E_\theta(x_i)\}$:
		* a set of points, not an analytic density (not smooth/continuous in a controlled way for sampling).
* A well-behaved latent space for naive sampling
	* Typical issues: 
		* Holes: unused regions
		* Folding/self-intersections: distant points on the manifold map close in latent 
		* Disconnected components: separate clusters with no smooth path 
		* Highly non-uniform density
	* So a random $z \in \mathbb R^d$ may decode off-manifold, and interpolation can cross invalid regions.
* A principled way to *sample*
	* Empirical sampling (pick a training $z_i$) is closer to memorization than generation.
	* Randomly sampling $z$ without a known prior is uncontrolled—yet we want $x \sim p_{\text{data}}$.

**Motivation for VAE**: impose a known prior and train encoded latents to match it.

VAE enables: $z \sim p(z) = \mathcal N(0, I) \quad\Rightarrow\quad x \sim p_\phi(x\mid z)$.

One common training objective (ELBO):

$$
\log p_\phi(x)\ \ge\ \mathbb E_{q_\theta(z\mid x)}[\log p_\phi(x\mid z)] - \mathrm{KL}(q_\theta(z\mid x)\,\|\,p(z)).
$$