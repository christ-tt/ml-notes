---
tags:
  - ML
  - Theory
  - MultiModal
  - Work
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
- Note: Manifold is defined in the ambient (data) space, not the latent space.

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
	* Around $z_0 = E_\theta(x_0)$, the decoder is approximately linear:$$D_\phi(z_0 + \delta) \approx D_\phi(z_0) + J_{D_\phi}(z_0)\,\delta,$$so small moves can correspond to semantic changes, while large moves can leave the region the decoder was trained on.
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
$$\log p_\phi(x)\ \ge\ \mathbb E_{q_\theta(z\mid x)}[\log p_\phi(x\mid z)] - \mathrm{KL}(q_\theta(z\mid x)\,\|\,p(z))$$


## How to Train an AE so that latent follows a specific distribution 
* When training the AE, explicitly impose the constraint that the hidden representation $z$ must follow a specific distribution, e.g. $P(z) \sim \mathcal N(0, I)$ .
* To generate novel values, sample $z$ from the prescribed distribution, and if it is properly sampled, the output should be a reasonable generation.
* ![[VAE-1.png]]

* Given encoder and decoder may have arbitrarily complex structure and their own parameters $\theta, \phi$, we ask, how to properly train an AE so that the latent follows, e.g. Normal?

### Prior of the Latent: Isotropic Gaussians
* $P(Z) = \mathcal N(0, I)$ 
* Maximally uninformative, simplest continuous prior with strong math properties
* The distribution is perfectly symmetric in every direction
	* 
	* No preferred direction, axis, and rotational invariance
	* Avoids arbitrary coordinate bias and special latent dimensions
* The distribution has independence of components
	* The different variables (sub-vectors $Z_1, Z_2$ of $Z$ ) are independent. i.e. $P(Z_1, Z_2) = P(Z_1)P(Z_2)$ 
	* Encourages disentanglement
	* Prevents latent dimensions from having to coordinate
		* The model can still learn correlated semantics, but it must encode them via the *decoder* , not via latent coupling
* Closure under marginalization
	* Each individually will also be isotropic: $P(Z_1) = \mathcal N(0, I)$ 
	* Making partial sampling, dimensional slicing, hierarchical models mathematically clean
* Analytic, closed-form, differentiable, and stable in high dimensions density, crucial for backprop gradients.
	* $$ P(Z) = \frac{1}{\sqrt{(2\pi)^d}} \exp \left (-0.5 |Z|^2 \right),$$ $$ -\log P(Z) = 0.5d\log 2\pi + 0.5 |Z|^2$$ 
	* The quadratic term $\to$ L2 penalty on latent magnitude, and by enforcing a Gaussian prior we penalize latent codes for drifting away from the origin.
* Training with statistical constraints:![[VAE-2.png]]
	* Minimize the error between $X, \hat X$ 
	* Minimize the KL divergence between the distribution of $z$ and the standard Gaussian $\mathcal N(0, I)$ 
		* By Maximum Likelihood, minimize the negative log likelihood of $z$ as computed from a standard Gaussian.

### Problem with Pushing latent to Zero
Minimizing negative log likelihood of latent $z$ for our MLE (MSE/KL loss), we are minimizing $$ \min \sum_z - \log \mathcal N(z; 0, I) = \min 0.5 \sum_x |z|^2 = \min \sum_X | E(X; \theta)| ^2$$
So, the objective now becomes $$ \min_{\theta, \phi} \sum_X |X-\hat X|^2 + \lambda |E(X; \theta)|^2 $$ Yet this simple formulation does not adequately capture the variation in the data. Pushing latent to **Zero**.

The generative portion of the model is just the decoder; the range of $z$ it accepts is still very small, around zero, and others are garbage.

The decoder can *only* generate data on a low-dimensional manifold of the space, of the same dimension as the input.
The decoder transforms the **planer** input space to a **curved** manifold in the output space, and the **Gaussian** to a  *non-Gaussian*.

The actual dimensionality of the data manifold may be (and generally will be) greater than the dimensionality of $z$.

Even if we capture the dimensionality of the principal manifold perfectly, there will almost alwasy be some variation






# Variational Auto Encoder

## **Important Unification & Comparison to Embedding Models**

- **Embeddings Are Just Manifold Compression**
    - Whether it is **ViT** (images), **BERT** (Text), or Word2Vec, the fundamental hypothesis is the **Manifold Hypothesis**
    - For images, the **Ambient Space**, raw input is, e.g. $ 256 \times 256 $, massive and mostly emptly
    - The **Manifold** lies on a low-dimensional surface emedded within that high-dimensional space;
    - Encoders $f_\theta$ maps the ambient space $u$ to the coordinate system of the manifold $z$ (embedding)
    - **Head** (Classifier, LLM...) is a function (linear / MLP) that operates on this manifold.
    - For **Intermediate Representation**, Embeddings are *not* the distribution parameters $\psi$, but just a transformed version of the input $u$; where we don't *explicitly* assume a distribution of the representation itself; Instead, we treat it as a *point mass*. The *distribution* comes at the *Head* part.
- **Transfer Learning**
    - We can decouple Encoders and Decoders
    - Given Encoders mainly handles *Manifold Mapping*, in image generation case, we can train Encoder + Decoder to reconstruct images, and then throw away the decoder to reuse encoder by freezing it, and slap a new *Classification Head* on top.
    - For our VAE case, we typically just take the mean $\mu$ and treat it as a determinisitc embeddings, ignoring variance $\sigma$
- **Unifying VAEs to Probabilistic Modeling**
    - Encoder is the determinisitc network, and we output the local parameters $\psi = [\mu, \sigma]$
    - The assumed distribution is Gaussian $z \sim \mathcal{N}(\mu, \sigma)$
    - **Sampling**: draw a latent vector $z$ using the reparameterization trick $z = \mu + \sigma \odot \varepsilon$
    - **The Prior** ($p(z)$) We assume the *global* distribution of latents across the whole dataset looks like a standard normal $\mathcal{N}(0, 1)$
    - The **Posterior** ($q(z\mid u)$): for a specific image, the distribution $\mathcal{N}(\mu_u, \sigma_u)$.
- **Stochastic Embeddings** of everything (e.g. LLM)
    - Called **Probabilistic Embedding** or **Bayesian Nueral Network**
    - Not the default
        - **Complexity**: sampling breaks standard backpropagtion, and we need *Reparameterization Trick** (moving the randomness to an external $\epsilon$
        - **Compute**: more parameters to predict both $\mu, \sigma$
        - **Necessity**: For discriminative tasks (Classification), we usually only cares about the best guess $\mu$, and the uncertainty $\sigma$ is often unnecessary.
    - Use it in:
        - Uncertainty Estimation
        - Diversity


## **Control** Problem: Why VAE?

- If a VAE just takes an image $u$ and reconstructs it $\hat u \approx u$, it's just a useless photocopier
- The **Real Purpose**: Generation. We don't train a VAE to use it on **existing** images. We train it so we can **throw away** the Encoder and **use the decoder alone**.
    - Training: Encoder to Dedcoder, learning to map data to the latent space
    - Inference: Random Noise $z \sim \mathcal{N}(0, 1) \to$ Decoder $\to$ New Image.
    - Result: The model *dreams* up a person who doesn't exist.
- Adding control: **CVAE**: making it like Stable Diffusion (with text prompt instructing to generate something).
    - Training:
        - We input $z$ + **Label**.
        - Input: $u$ (Image of digit 7) + $c$ (Lable 7)
        - Latent: Encoder maps image to $z$
        - Decoder: Taking $(z, c)$ and tries to reconstruct.
    - Inference:
        - Sample noise $z$
        - Inject Control: pick label $c = '9'$
        - Decoder outputs a generated 9.
- In LLaVA, the 'Control' is the image embedding
    - The LLM decoder is generating text, conditioned on the image latent.
    - Conceptually identical to a CVAE, where label $c$ is replaced by a rich vector representation from a ViT.



## Questions

- For image embedding models (like ViT), or in general for all embedding models (like using BERT for text embeddings, or even LLM for word embeddings), are we essentially using the same ideas for Auto-Encoders: that all data lie in the smaller manifold compared to the ambient space? Like a compression, we enforce the representation, our 'encoded' way of input from the original, raw input, will be better suited for down-stream tasks, like adding a LM head or MLP for certain classification work, where we then assume a certain (categorical) distribution?

- Following this question, even we name 'intermediate representation' as part of the whole model, can we effectively assume encoder and decoders are decoupled? Where the embedding of the encoder can be used, or easily fine-tuned, for other down-stream tasks? Does this apply to AE, or VAE's encoder and decoder as well?

- Can we unify the way VAE is doing encoding, with how we form our 'stochastic' part of the model? i.e. we are outputing $\psi$ with local parameters for the distribution, whereas in VAE for the latent variables we are essentially outputting mean and var for a guassian distribution? When we say our latent does not follow a specific distribution, we are saying the local parameters themselves? 

- In VAE, we say given the input, we have the latent from the encoder, where it is local parameters of a distribution; can we apply this to ALL the reprensentation / embeddings for other tasks, e.g. language modeling? Instead of output a fixed vector for a word, why don't we follow the latent idea of VAE and train the encoder in a way that it output a parameters of the distribution? Adding the stochastic even on the fixed embedding vectors.  

- Taking a step back, a more fundamental question is, what is the purpose of VAE? In image understanding, we have both text and image input, where the image going to ViT for an embedding, and we also have text embedding from our decoder's embedding table, and then we do certain cross-attention and the decoding part will be 'controlled', (and for image generation as well). For VAE, however, we input an image, and we are output an image as well, there's no explicit control in it, is it? The best we can do is outputting the exact same image? Why we need VAE, or in general, this task at all? How do we add 'control' in the images we are generating?





