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
* A projection operator onto the data manifold
* Good reconstructions
* Meaningful latent directions *locally*

### Incapability
* A *pabability distribution* over the manifold
* A well-behaved latent space
* A pricipled way to *sample* 

## VAE: probabilistic latent-variable model + variational training objective
Choose a prior $p(z)$ (commonly $\mathcal N(0,I)$) and a decoder likelihood $p_\phi(x \mid z)$. Since $p_\phi(z \mid x)$ is intractable, introduce an encoder $q_\theta(z \mid x)$ and maximize the evidence lower bound (ELBO):

$$
\log p_\phi(x)
=
\underbrace{\mathbb E_{q_\theta(z\mid x)}[\log p_\phi(x\mid z)] - \mathrm{KL}(q_\theta(z\mid x)\,\|\,p(z))}_{\mathcal L_{\text{ELBO}}(x)}
\ +\ \mathrm{KL}(q_\theta(z\mid x)\,\|\,p_\phi(z\mid x))
$$

so

$$
\log p_\phi(x) \ge \mathcal L_{\text{ELBO}}(x)
$$

Intuition:
- $\mathbb E_{q_\theta(z\mid x)}[\log p_\phi(x\mid z)]$: reconstruction term (likelihood)
- $\mathrm{KL}(q_\theta(z\mid x)\,\|\,p(z))$: regularizes latents to match the prior (enables sampling $z \sim p(z)$ for generation)

(Common Gaussian encoder uses the reparameterization trick: $z=\mu_\theta(x)+\sigma_\theta(x)\odot \epsilon,\ \epsilon\sim\mathcal N(0,I)$.)