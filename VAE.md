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
* A manifold *parameterization*
	* To parameterize a manifold $\mathcal M \subset \mathbb R^D$ means:
		* Assign each data point $x \in \mathcal M$ a *coordinate* $z \in \mathbb R^d$ 
		* Such that nearby points on $\mathcal M$ have nearby coordinates
	* AE learns two maps
		* $E: \mathcal M \to \mathbb R^d$, assigns *coordinates* to points on the manifold
		* $D: \mathbb R^d \to \mathbb R^D$ , maps *coordinates* back to *ambient* space
		* with the constraint $D(E(x)) \approx x, \forall x \in \mathcal M$ 
	* i.e. charts work in differential geometry, and generally a **local parameterization**, not guaranteed to be globally **invertible**. 
* A projection operator onto the data manifold: $P(x) := D(E(x))$ 
	* For $x \in \mathcal M$: $P(x) \approx x$
	* For $x \notin \mathcal{M}$: $P(x) \approx$ nearest point on $\mathcal M$.
	* Gemetrically:
		* Encoder removes off-manifold components
		* Decoder reconstructs the closet on-manifold point
	* So $P$ behaves like a nonlinear projection onto the manifold
* Good reconstructions
	* $x \to z \to \hat x$ 
	* Mapping data points on the manifold, to latent coordinates, back to ambient space
* Meaningful latent directions *locally*
	* Latent direction: a direction $v \in \mathbb R^d$ such that $D(z + \epsilon v)$ produces a meaningful change in data space (e.g. rotate a face, change lightning, modify pitch)
	* Locally as AE only guarantees correctness near latent codes it has seen, not on the whole latent space.
	* Around $z_0 = E(x_0)$, $D(z_0 + \delta) \approx x_0 + J_D(z_0)\delta$ where
		* $J_D(z_0)$ spans tangent space directions
		* This approximation breaks down globally
		* So small moves means semantic change, and large moves collapse or nonsense.
	* AE latent spaces typically have *Folds, Self-Intersections, Holes, Disconnected Regions*.
		* One latent direction near one sample may mean something entirely different elsewhere.
### Incapability
* A *probability distribution* over the manifold
	* We want to switch from a **point mapping** to a **distribution** 
	* In vanilla AE, encoder $E(x) = z$ is a single deterministic vector, and decoder $D(z) = \hat x$, so there's no probability distribution in the model.
	* Implicitly, we have an empirical distribution of latent codes: $\{z_i = E(x_i)\}$ 
		* But, this is just a set of points, not a density, not continuous, not smooth, not known anlalytically
	* We want a probability measure $p_{\mathcal M}(x)$ supported on $\mathcal M$ , equivalently: $p(z)$ such that $x = D(z)$.
		* answering which regions of the manifold are likely, which are rare, and how often should we sample each 'mode', where AE can't answer.
* A well-behaved latent space, where AE has
	* Holes: large regions of latent space never used, sampling could land in invalid zones
	* Folds / self-intersections: distant points on the manifold map close in latent, introducing decoder ambiguity
	* Disconnected regions: separate clusters with no smooth path, so intepolation crosses invalid space
	* Non-uniform density: some areas densely packed and others sparse, with sampling bias unknown.
	* Well-behaved: 
		* A known prior $p(z)$
		* Smooth, continuous support, no holes, consistent neighborhoods.
* A principled way to *sample* 
	* AE doesn't catch distribution, empirical sampling (pick a random training z) just memorizes
	* AE degenerates sampling to nearest-neighbor reconstruction, interpolation between known points, adding untrolled noise, where we need $x\sim p_{data}$ 

VAE enables: $z \sim p(z) = \mathcal{N}(0, I) \quad\Rightarrow\quad x = D(z)$  