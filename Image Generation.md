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
$$
\log p_\phi(x)\ \ge\ \mathbb E_{q_\theta(z\mid x)}[\log p_\phi(x\mid z)] - \mathrm{KL}(q_\theta(z\mid x)\,\|\,p(z))
$$


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
Minimizing negative log likelihood of latent $z$ for our MLE (MSE/KL loss), we are minimizing 
$$ 
\min \sum_z - \log \mathcal N(z; 0, I) = \min 0.5 \sum_x |z|^2 = \min \sum_X | E(X; \theta)| ^2
$$
So, the objective now becomes $$ \min_{\theta, \phi} \sum_X |X-\hat X|^2 + \lambda |E(X; \theta)|^2 $$ Yet this simple formulation does not adequately capture the variation in the data. Pushing latent to **Zero**.

The generative portion of the model is just the decoder; the range of $z$ it accepts is still very small, around zero, and others are garbage.

The decoder can *only* generate data on a low-dimensional manifold of the space, of the same dimension as the input.
The decoder transforms the **planer** input space to a **curved** manifold in the output space, and the **Gaussian** to a  *non-Gaussian*.

The actual dimensionality of the data manifold may be (and generally will be) greater than the dimensionality of $z$.

Even if we capture the dimensionality of the principal manifold perfectly, there will almost alwasy be some variation


# **Variational Auto Encoder**

## **Training (The Encoder-Decoder Loop)**
**Goal**: Learn the manifold parameters $\theta$ (Decoder) and the mapping $\phi$ (Encoder).
1. **Input** ($x$): A real image (e.g., a handwritten "7").
2. **Encoder** ($f_{\text{enc}}$), parametrized by $\phi$:
    - Maps $x$ to latent parameters: Mean $\mu_z$ and Log-Variance $\log \sigma_z^2$.
    - $\psi_{\text{enc}} = [\mu_z, \sigma_z]$.
    - Define the **Approximate Posterior** $q(z \mid u) = \mathcal{N}(\mu_z, \sigma_z)$.
        - Estimate **True Posterior** $p(z \mid u)$ latent codes over ALL $u$, intractable.
3. **Reparameterization** (The Stochastic Trick):
    - We want to sample $z \sim q(z \mid u)$
    - We cannot backpropagate through a random node $z \sim \mathcal{N}(\mu, \sigma)$.
    - **Trick**: Sample explicit noise $\epsilon \sim \mathcal{N}(0, I)$ and shift it:
    $$
    z = \mu_z + \sigma_z \odot \epsilon
    $$
    - $\odot$ is the Hadamard Product, i.e. **element-wise multiplication**.
    - Now gradients can flow back to $\mu$ and $\sigma$.
    - This is simply the formal way of saying 'Sampling from Gaussian is just adding noise'.
4. **Decoder** ($f_{\text{dec}}$), parameterized by $\theta$:
    - Maps $z$ back to data space parameters (usually just the mean).
    - $\psi_{\text{dec}} = \mu_x$, define the **Likeihood** $p(x\mid z) = \mathcal{N}(\mu_x, I)$
    - $\hat{x} = \text{Decoder}(z)$.
5. **Loss Calculation** (ELBO):
    - Term 1 (**Reconstruction**): How close is $\hat{x}$ to $x$? (MSE Loss).
    - Term 2 (**Regularization**): How close is the approximate posterior $\mathcal{N}(\mu_z, \sigma_z)$ to the prior $\mathcal{N}(0, I)$? (KL Divergence).
    $$
    \mathcal{L} = \|x - \hat{x}\|^2 + \beta \cdot D_{KL}(\mathcal{N}(\mu_z, \sigma_z) \| \mathcal{N}(0, I))
    $$

## **ELBO (Evidence Lower Bound)**
The ELBO is the objective function we actually maximize when training a VAE.Since we cannot calculate the true probability of the data (the "Evidence" $\log p(u)$) directly because the integral over all latents is intractable, we optimize this proxy instead.

From Jensen's Inequility:
$$
\text{ELBO} = \mathbb{E}_{q} \left[ \log \frac{p(x, z)}{q(z \mid x)} \right]
$$
We essentially have,
$$\begin{align}
\text{ELBO} &= \mathbb{E}_{q} \left[ \log \frac{p(x \mid z) p(z)}{q(z \mid x)} \right] \\
&= \mathbb{E}_{q} \left[ \log p(x \mid z) + \log \frac{p(z)}{q(z \mid x)} \right] \quad \text{(Log product rule)} \\
&= \underbrace{\mathbb{E}_{q} [\log p(x \mid z)]}_{\text{Reconstruction}} + \mathbb{E}_{q} \left[ \log \frac{p(z)}{q(z \mid x)} \right] \\
\mathbb{E}_{q} \left[ \log \frac{p(z)}{q(z \mid x)} \right] &= - \mathbb{E}_{q} \left[ \log \frac{q(z \mid x)}{p(z)} \right] = - D_{KL}(q(z \mid x) \| p(z))
\end{align}$$
Thus we have
$$
\text{ELBO} = \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x \mid z)]}_{\text{Reconstruction Term}} - \underbrace{D_{KL}(q_\phi(z|x) \,\|\, p(z))}_{\text{Regularization Term}}
$$

The marginal log-likelihood (the Evidence) can be decomposed into:
$$
\log p_\theta(u) = \text{ELBO} + D_{KL}(q_\phi(z|x) \,\|\, p_\theta(z|x))
$$
Since KL Divergence is always non-negative ($D_{KL} \ge 0$), we get the inequality:
$$  
\log p_\theta(u) \ge \text{ELBO}
$$

### **Reconstruction Term**
- **Goal**: Maximize the likelhood of the real image $u$ given the latent $z$.
- **Action**: We want latent code $z$ to be very specific and distinct for every image, so it can reconstruct details perfectly. Ideally, it wants $q(z \mid u)$ to be a Dirac delta (point mass) exactly at the perfect code.
- **Effect**: Pushing the variance $\sigma^2 \to 0$.

1. The Geometric Interpretation (MSE)
    the reconstruction loss is almost always Mean Squared Error (MSE):
    $$
    \text{Loss} = \| u - \hat{u} \|^2 = \sum (u_i - \hat{u}_i)^2
    $$
    Interpretation: "Minimize the physical distance between the pixel values of the original image $u$ and the generated image $\hat{u}$.", pulling the "Arrow" ($\hat{u}$) closer to the "Target" ($u$) in Euclidean space.
2. The Probabilistic Interpretation (Log-Likelihood)
    We assumed the likelihood is a Gaussian with fixed variance $\sigma=1$:
    $$
    p_\theta(u \mid z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} (u - \hat{u})^2}
    $$
    If we take the Logarithm of this (which we do for Maximum Likelihood Estimation):
    $$
    \log p_\theta(u \mid z) = \underbrace{\log \left( \frac{1}{\sqrt{2\pi}} \right)}_{\text{Constant}} - \underbrace{\frac{1}{2} (u - \hat{u})^2}_{\text{Proportional to -MSE}}
    $$
3. The Equivalence$$\text{Maximizing Log-Likelihood} \equiv \text{Minimizing MSE}$$So, when we say "Maximize the probability of the image," we are literally saying "Minimize the pixel distance between the input and the output."

The expansion of the expectation term into its integral form:
$$
\mathbb{E}_{z \sim q_{\phi}(z \mid u)} [\log p_\theta(u \mid z)] = \int_{-\infty}^{\infty} q_{\phi}(z \mid u) \log p_\theta(u \mid z) \, dz
$$

- The Weight ($q_{\phi}(z \mid u)$): This is the probability density of the latent code $z$ predicted by the Encoder. It tells us how much "weight" to assign to specific regions of the latent space (e.g., the region around the mean $\mu$).
- The Score ($\log p_\theta(u \mid z)$): This is the reconstruction log-likelihood (the negative MSE) for a specific $z$.
- The Operation ($\int$): We are summing up the reconstruction scores for every possible latent code, weighted by how likely the Encoder thinks that code is.
This integral is effectively calculating the **weighted average reconstruction quality** over the entire "cloud" of latent codes predicted by the Encoder.

### **Regularization Term**

- **Goal**: Minimize the divergence between the encoder's output $q(z \mid u)$ and the prior $p(z) = \mathcal{N}(0, I).
- **Action**: We want the latent codes to be fuzzy, overlapping standard Gaussians, forbitting the model from cheating by memorizing specific points.
- **Effect**: Pushing mean $\mu \to 0$ and variance $\sigma^2 \to 1$.

### **Approximate Posterior v.s. True Posterior**

**Indexing**: we index **True Posterior** $p(z \mid u)$ with $\theta$ because the 'Truth' is defined by the **Decoder**. As we update $\theta$, we change the mapping from latent to image.
- The Target ($u$): The Real Image. (Fixed).
- The Archer ($\theta$): The Decoder.
- The Shot ($z$): The latent code provided by the Encoder.
- The Arrow Landing Spot ($\hat{u}$): The Reconstruction.
- The Likelihood $p_\theta(u \mid z)$ is the Score. It measures the distance between the Arrow ($\hat{u}$) and the Target ($u$).
$$
\text{Log Likelihood} \propto - \| \text{Target}(u) - \text{Arrow}(\hat{u}) \|^2
$$
So, $p_\theta(u \mid z)$ is the value we maximize. We do this by moving the Arrow ($\hat{u}$) closer to the Target ($u$). $\hat u = f_\theta(z), u \sim \mathcal{N}(\hat u, I)$, i.e. $p_\theta(u \mid z) = p(u \mid \psi) = \mathcal{N}(u; \hat u, \sigma^2I)$

**Goal**: Show that minimizing the KL divergence between the Approximate Posterior $q_\phi(z \mid u)$ and the True Posterior $p_\theta(z \mid u)$ is equivalent to maximizing the ELBO (Evidence Lower Bound).

$$
\begin{align}
\mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p_\theta(z \mid u)\big) &= \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{q_\phi(z \mid u)}{p_\theta(z \mid u)} \right] \\
&= \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} [\log p_\theta(z \mid u)]
\end{align}
$$
Apply Bayes' Rule to the true posterior: $p_\theta(z \mid u) = \frac{p_\theta(u \mid z) p(z)}{p_\theta(u)}$.
$$
\begin{align}
\dots &= \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{p_\theta(u \mid z) p(z)}{p_\theta(u)} \right] \\
&= \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} [\log p_\theta(u \mid z)] - \mathbb{E}_{z \sim q_\phi} [\log p(z)] + \underbrace{\mathbb{E}_{z \sim q_\phi} [\log p_\theta(u)]}_{\log p_\theta(u) \text{ is const w.r.t } z}
\end{align}
$$
Now, rearrange the terms to group the KL Divergence to Prior and the Reconstruction Loss:
$$
\begin{align}
\mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p_\theta(z \mid u)\big) &= \underbrace{\left( \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} [\log p(z)] \right)}_{\mathrm{KL}(q_\phi(z \mid u) \,\|\, p(z))} - \mathbb{E}_{z \sim q_\phi} [\log p_\theta(u \mid z)] + \log p_\theta(u)
\end{align}
$$
Thus, the relationship is:
$$
\log p_\theta(u) = \underbrace{\mathbb{E}_{z \sim q_\phi} [\log p_\theta(u \mid z)] - \mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p(z)\big)}_{\text{ELBO}(\theta, \phi)} + \underbrace{\mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p_\theta(z \mid u)\big)}_{\ge 0}
$$

Since $\log p_\theta(u)$, the **evidence**, is **fixed** for a given data point, maximizing ELBO is mathematically equivalent to minimizing the divergence between approxmiate posterior and the true posterior
$$\begin{align}
\phi^*, \theta^* &= \arg \max_{\phi, \theta} \text{ELBO}(u) \\
&= \arg \max_{\phi, \theta} \left( \mathbb{E}_{z \sim q_\phi} [\log p_\theta(u \mid z)] - \mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p(z)\big) \right)
\end{align}$$

The first term is **reconstruction**, maximizing likelihood of data given latent, and the second term is **regularization**, forcing approximate posterioir $q$ to be close to prior $p(z)$


## **Inference (The Generative Loop)**

**Goal**: Generate new data. The Encoder is deleted.
1. **Input** ($z$): We need a seed.
    - Sample $z \sim \mathcal{N}(0, I)$ (The Prior).
2. **Decoder** ($f_{\text{dec}}$):
    - $\psi_{\text{out}} = \text{Decoder}(z)$.
    - (Typically $\psi_{\text{out}}$ is just the mean image $\mu_x$).
3. **Output** ($x_{\text{gen}}$):
    - Usually, we just display the mean $\mu_x$ directly (it looks cleaner).
    - Strictly speaking, we should sample $x_{\text{gen}} \sim \mathcal{N}(\mu_x, C)$
        - For Gaussian likelihood, we usually use **additive** noise
        $$
        x = D(z; \phi) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, C)
        $$
        which is mathematically equivalent to
        $$
        p_\theta(x \mid z) = \mathcal{N}(x; \mu=D(z), \Sigma=C)
        $$
        - To decide variance $C$,
        $$
        \log p(x \mid z) \propto -\frac{1}{2} (x - D(z))^T C^{-1} (x - D(z)) - \frac{1}{2} \log |C|
        $$
        We could
            - Learn it (Heteroscedastic): The Decoder outputs both the mean $\mu_x$ and the variance $\sigma_x^2$ for every pixel.
                - Pros: The model learns which parts of the image are detailed (low variance) vs. noisy/texture (high variance).
                - Cons: Unstable training. It often collapses to predicting zero variance (infinite likelihood).
            - Fix it (Homoscedastic) \[**standard**\]
                - We simply assume $C = \sigma^2 I$ is a fixed scalar constant (hyperparameter) for all data points.
                - If $C = I$ (identity matrix), the term $\log |C|$ becomes constant and vanishes from the optimization.
                - The likelihood term simplifies purely to:
                $$
                \mathcal{L}_{\text{recon}} \propto - \| x - D(z) \|^2
                $$
                - Result: This is why VAEs are trained with MSE Loss. MSE is just the negative log-likelihood of a Gaussian with fixed variance.



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


---

# **Hierarchical VAE $\to$ Diffusion**

To understand Diffusion, we don't need to learn a "new" algorithm; we just need to take a **Hierarchical VAE (HVAE)** and take it to its logical extreme.

An HVAE is simply a VAE that doesn't stop at one layer of latent variables. Instead of a single compression step, it performs a sequence of them.

**Motivation**: Trying to compress a complex $256 \times 256$ image into a single vector $z$ causes "information bottleneck." The model is forced to average out fine details (high frequency) to save the global structure (low frequency).

## **Architecture**
### **A. The Encoder (Bottom-Up / Inference)**

We compress the data step-by-step.

* **Step 1:** Map Image $x$ to $z_1$ (Low-level features, like edges).
* **Step 2:** Map $z_1$ to $z_2$ (Mid-level shapes).
* **Step T:** Map $z_{T-1}$ to $z_T$  (High-level semantics, "Cat").
$$
q(z_{1:T} \mid x) = q(z_1 \mid x) \times q(z_2 \mid z_1) \times \dots \times q(z_T \mid z_{T-1})
$$
We are assuming a **Markov Chain**: $z_{t}$ only depends on $z_{t-1}$ for $t > 1$.

### **B. The Decoder (Top-Down / Generation)**

We reconstruct the image step-by-step.

* **Step 1:** Sample abstract concept $z_T$.
* **Step 2:** Flash out details to get $z_{T-1}$.
* **Final Step:** Generate pixels $x$ from $z_1$.
$$
p(x, z_{1:T}) = p(z_T) \times p(z_{T-1} \mid z_T) \times \dots \times p(x \mid z_1)
$$

---

## **ELBO**

$$
\begin{align}
\text{ELBO} &= \mathbb{E}_{q} \left[ \log \frac{p(x, z_{1:T})}{q(z_{1:T} \mid x)} \right] \\
\frac{p(x, z_{1:T})}{q(z_{1:T} \mid x)} &= \frac{p(x \mid z_1) \, p(z_1 \mid z_2) \dots p(z_{T-1} \mid z_T) \, p(z_T)}{q(z_1 \mid x) \, q(z_2 \mid z_1) \dots q(z_T \mid z_{T-1})} \\
\text{ELBO} &= \underbrace{\mathbb{E}_q [\log p(x \mid z_1)]}_{\text{Reconstruction}} + \underbrace{\sum_{t=2}^{T} \mathbb{E}_q \left[ \log \frac{p(z_{t-1} \mid z_t)}{q(z_{t-1} \mid z_{t-2})} \right]}_{\text{Transition Terms}} - \underbrace{D_{KL}(q(z_T \mid z_{T-1}) \| p(z_T))}_{\text{Top-Level Prior}} \\
\end{align}
$$
where $z_0 = x$.
Look at the indices in the middle term.
- The Decoder moves down: $p(z_{t-1} \mid z_t)$.
- The Encoder moves up: $q(z_t \mid z_{t-1})$.
- The Mismatch: Unlike Vanilla VAE, the distributions in the middle don't perfectly pair up into neat KL divergence terms like $KL(q \| p)$ because they are conditioned on different things. This makes standard HVAEs tricky to optimize.

---

## **Diffusion Trick - Inverting Encoder**

Note that, given the ELBO Term:
$$
\log \frac{p(z_{t-1} \mid z_t)}{q(z_t \mid z_{t-1})}
$$
We **can't** simply negate this to form a KL divergence:
$$
- \log \frac{q(z_t \mid z_{t-1})}{p(z_{t-1} \mid z_t)}.
$$
This is NOT a valid KL Divergence. KL Divergence $D_{KL}(A \| B)$ requires $A$ and $B$ to be distributions over the same variable.Here, $q$ gives probabilities for $z_t$, but $p$ gives probabilities for $z_{t-1}$.

To make the math clean (and to unlock Diffusion Models), we perform a trick: rewritting the ELBO using Bayes' rule on the Posterior side, so that we can compare the encoder directly to the decoder. To do this, we must force the Encoder to *look "backward"* (from $t$ to $t-1$), just like decoder.

**Goal**: Convert the forward rule $q(z_t \mid z_{t-1})$ (Bottom-Up) into a Reverse Posterior $q(z_{t-1} \mid z_t, x)$.
- Idea: "Given that I'm at the noisy state $z_t$, and I know the original image $x$, what was the previous step $z_{t-1}$?"

**Derivation**: Using Bayes' Rule:
$$
q(z_{t-1} \mid z_t) = \frac{q(z_t \mid z_{t-1}) \, q(z_{t-1})}{q(z_t)}
$$

Why condition on $x$? The terms $q(z_{t-1})$ and $q(z_t)$ are marginal probabilities. In a Markov chain starting from data $x$, these marginals depend entirely on the starting point $x$.
- Without $x$, $q(z_t)$ is a mixture over the entire dataset (very complex).
- With $x$, the math becomes solvable (Gaussian).
$$
\begin{align}
q(z_{t-1} \mid z_t, x) &= \frac{q(z_{t-1}, z_t, x)}{q(z_t, x)} \\
&= \frac{q(z_t \mid z_{t-1}, x)q(z_{t-1}, x)}{q(z_t \mid x)q(x)} \\
&= \frac{q(z_t \mid z_{t-1}) \, q(z_{t-1} \mid x)}{q(z_t \mid x)}
\end{align}
$$

We defined the forward process as **Gaussian Noise**:
1. $q(z_t \mid z_{t-1})$ is Guassian (Step definition).
2. $q(z_t \mid x)$ is Gaussian (Sum of Gaussians is Gaussian).
3. $q(z_{t-1} \mid x)$ is Gaussian.
Since we are dividing/multiplying Gaussians, the result $q(z_{t-1} \mid z_t, x)$ is also a Gaussian.

This allows us to write the ELBO term as a valid KL Divergence:
$$
D_{KL} \Big( \underbrace{q(z_{t-1} \mid z_t, x)}_{\text{Encoder Reverse (Truth)}} \;\|\; \underbrace{p_\theta(z_{t-1} \mid z_t)}_{\text{Decoder Reverse (Prediction)}} \Big)
$$

Now we can rewrite the ELBO using this formulation (which is standard in Diffusion math), the sum becomes a beautiful chain of KL divergences:
$$
\text{ELBO} = \underbrace{\mathbb{E}[\log p(x \mid z_1)]}_{\text{Reconstruction}} - \sum_{t=2}^T \underbrace{D_{KL}\Big( q(z_{t-1} \mid z_t, x) \,\|\, p(z_{t-1} \mid z_t) \Big)}_{\text{Denoising Matching}} - \underbrace{D_{KL}(q(z_T \mid x) \| p(z_T))}_{\text{Prior Matching}}
$$

This equation is the foundation of Diffusion Models. It says: "Make your learned reverse step $p$ match the true mathematical reverse step $q$."

#### **More on Motivation of Revsered Posterior - Compute Persepective**
Calculating $q(z_{t-1} \mid z_t)$ is normally impossible.
Think about a generic noisy image $z_t$ (pure static).
- Question: "What was the previous step $z_{t-1}$?"
- Answer: "I have no idea. It could have been any image in the universe that got corrupted."

Mathematically, to solve this, you would have to **integrate over all possible images** in the universe:
$$
q(z_{t-1} \mid z_t) = \int q(z_{t-1} \mid z_t, x) \, p(x) \, dx
$$
This is intractable. We cannot sum over all possible photos.

**Solution**: Conditioning on $x_0$.
We know the original image $x_0$ (the ground truth from our dataset)
If I tell you:
    - Current State: "This specific noisy blob."
    - Destination: "This specific photo of a Cat."
Now, the question "Where did I come from?" has a specific, calculable answer. We don't need to consider all images, just the **path** from this Cat to this Noise.
This makes the Reverse Posterior Tractable.

Since our noise process is Gaussian, this specific path is just a weighted average of the current noise and the original image.

---

### **VAE to Diffusion**

Imagine a specific type of HVAE with **three constraints**:

1. **Infinite Depth:** We add more and more layers ($T\to \infty$).
2. **Same Dimension:** Every latent layer $z_t$  has the **same shape** as the image $x$ (no compression in size, only in information).
3. **Fixed Encoder:**:
    * In HVAE, we *learn* the Encoder $q_\phi$.
    * In this special case, we **fix** the Encoder to be a simple, non-learnable noise injector.
    * $q(z_t \mid z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta} z_{t-1}, \beta I)$. (Just adding Gaussian noise).

**Then, the HVAE becomes a Diffusion Model.** Mentally,

- **HVAE:** "I will learn a smart hierarchy of features to compress the image."
- **Diffusion:** "I will define the 'encoding' simply as destroying the image with noise layer-by-layer. Then, I will treat the 'decoding' as a massive HVAE that learns to reverse this destruction."

So, mathematically, a Diffusion model **is** an HVAE where the inference path is fixed to be a noise process, and we only train the generative path to undo it.


| Feature | **Standard HVAE** | **Diffusion Model (VDM)** |
| --- | --- | --- |
| **Latent Layers** | Several ($T \approx 5 \sim 10$) | Many ($T \approx 1000$) |
| **Latent Dim** | Getting smaller (Compressed) | Same size as Image (Full Res) |
| **Encoder $q(z \mid x)$** | **Learned** Neural Network | A fixed Linear Schedule (solver), not trainable
| **Decoder $p(x \mid z)$** | Learned Neural Network | A Neural Network (U-Net)
| **Latent Meaning** | Abstract Features (Edges, Shapes) | Noisy Images (Pixel soup) |


---

# **Diffusion (DDPM)**

A Diffusion Model is a **Parameterized Markov Chain** trained using variational inference. It can be understood as a Hierarchical VAE with $T \to \infty$ layers, where the encoder is fixed and the latent variables have the same dimension as the input.

It consists of two processes:
1. Forward Process (Diffusion): A fixed, linear chain that gradually destroys structure in data $x_0$ by adding noise until it becomes pure Gaussian noise $x_T$.
2. Reverse Process (Denoising): A learned chain that attempts to invert the diffusion process, restoring structure from noise.

Key **Assumptions**:
1. Markov Property: The future state depends only on the current state.
2. Gaussian Transitions: The noise added at each step is Gaussian. This allows us to sum variances easily.
3. Small Steps ($T \to \infty$): The noise added at each step is small enough that the reverse distribution $p(x_{t-1}|x_t)$ can also be approximated as Gaussian.


## **Forward Process - Fixed Encoder**

Instead of training an encoder, we define a fixed **Variance Schedule** $\beta_1, \dots, \beta_T$ (scalars) that controls the noise level at each step with **Linear Scheduler**.

The transition probability is defined as
$$
q(x_t \mid x_{t-1} = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_tI\right)
$$
- The **Jump** Property: Because Gaussian distributions are additive, we do not need to iterate step-by-step during training. We can sample $x_t$ directly from $x_0$ in closed form. Let $\alpha_t = 1-\beta_t$ and $\bar \alpha_t = \prod_{i=1}^t \alpha_i$:
$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar \alpha_t}x_0, (1-\bar \alpha_t)I\right)
$$
- **Implementation** (reparameterization): to sample $x$, we simply scale the image and add noise
$$
x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1 - \bar \alpha_t}\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

### **Linear Scheduler**

Defines how much noise is added at every single step $t$. We define a start value $\beta_1$ and an end value $\beta_T$. The scheduler calculates the variance $\beta$ for every step $t$ using a simple line equation
$$
\beta_t = \beta_1 + \frac{t}{T}(\beta_T - \beta_1)
$$
Standard values: we use $\beta_1 = 0.0001$ as step 1 adds tiny noise, and $\beta_T = 0.02$ adding roughly 200 times more noise.

**Motivation**: if we use constant noise, either it will be
- too aggressive: perhaps the image would turn to pure static by step 10, and the remaining 990 steps would just be mixing static with static, wasting compute
- too weak: at step 1000, the image would still be visible ghostly, breaking the assumption that $x_T$ is pure Gaussian noise.

Linear schedule ensures a smooth, gradual decay of the **Signal-to-Noise Ratio (SNR)**.
-
---

## **Reverse Process - Learned Decoder**

Since the forward process destroys information, the reverse step $q(x_{t-1} \mid x_t)$ is intractable at inference time as we don't know which specific $x_0$ the noise came from; but during training, we can still condition it on $x_0$.
We train a neural network $p_\theta$ to approximate $q(x_{t-1} \mid x_t)$.
$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$
- Input: Noisy image $x_t$, time step $t$;
- Output: Predicted mean $\mu_\theta$. (Variance $\Sigma_\theta$ is typically fixed to $\beta_t$)


---

## **Objective - Loss Function**

### **A. Teacher vs. Student**
- The Teacher (Reverse Posterior): $q(x_{t-1} \mid x_t, x_0)$.
    - This distribution knows the ground truth $x_0$ and calculates the exact step required to move from $x_t$ towards $x_0$.
- The Student (Model): $p_\theta(x_{t-1} \mid x_t)$.
    - This network only sees the noise $x_t$ and must guess the direction.

### **B. The Simplified Loss**
Matching the means of these distributions is mathematically equivalent to predicting the noise $\epsilon$ that was added.
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\underbrace{\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon}_{\text{Noisy Input } x_t}, t) \|^2 \right]$$Task: "Here is a noisy image. Tell me what the noise looks like."


---

## **The Diffusion Transformer (DiT)**

While U-Nets were the standard, modern architectures (Sora, SD3) use Vision Transformers (DiT) for better scaling.

**Workflow**
1. Patchify (Tokenizer):
    - The noisy input $x_t$ (e.g., $32 \times 32 \times 4$ latent) is chopped into patches (e.g., $2 \times 2$), resulting in a sequence of tokens.
2. Context Injection (adaLN):
    - Standard ViTs use LayerNorm. DiT uses Adaptive Layer Norm (adaLN).
    - The Time $t$ and Class/Text Label $c$ are embedded into a vector.
    - This vector predicts the **Scale** ($\gamma$) and **Shift** ($\beta$) parameters for the normalization layers inside the transformer block.
    - *Effect*: The time step globally modulates the activations of the network (e.g., "Shift activations heavily for high noise").
3. Transformer Blocks:
    - Standard Multi-Head Self-Attention (Global context) and Pointwise MLPs.
4. Un-Patchify:
    - Linear projection maps tokens back to the original pixel/latent space to output the predicted noise $\epsilon_\theta$.

**DiT vs. U-Net**

| Feature | U-Net | DiT (Transformer) |
| --- | --- | --- |
| **Inductive Bias** | **Local (CNN)**. Great for textures/edges | **GLobal (Attention)**. Better for sematntic structure. |
| **Conditioning** | Added to residual blocks via Cross-Attention | Modulates LayerNorm params (adaLN) |
| **Scaling** | Difficult to scale depth/width | Predictable Scaling Laws (compute $\propto$ performance) |

---

## **Inference - Sampling Algorithm**

To generate an image, we start from pure noise and reverse the chain using the trained model.

**DDPM Sampling**:
1. Start: Sample $x_T \sim \mathcal{N}(0, I)$.
2. Loop: For $t = T, T-1, \dots, 1$:
    - Use DiT to Predict Noise:
        - input time step $t$ and current noisy image $x_t$
        - DiT engine patchify, embed context $(\gamma, \beta)$ for adaLN
        - Outputs: $\hat \epsilon = \epsilon_\theta(x_t, t)$.
    - Denoise (Remove predicted noise):
    $$
    \mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar \alpha_t}}\hat\epsilon\right)
    $$
    - Add Langevin Noise: (Crucial for correct texture/diversity).
    $$
    x_{t-1} = \mu_t + \sigma_t z, \qquad z \sim \mathcal{N}(0, I)
    $$
3. End: $x_0$ is the generated image.

*Note*: Model samplers (DPM-Solver) treat this loop as an ODE and can skip steps, reducing inference from 1000 steps to $\sim$ 20.

---

## **Advanced Sampling: ODE / SDE**

### **The Math of Speed: Probability Flow ODE**

Intuition: Imagine a cloud of particles drifting and diffusing.
- SDE (The Drunk Walk): Pick one particle. It jitters around randomly. It eventually hits the target distribution.
- ODE (The Smooth Stream): Imagine the "average flow" of the particles. If you trace a line that follows the density gradient of the cloud, you get a smooth, non-random curve.

**Why does this matter?**
- Random paths are jagged. To trace a jagged line accurately, you need tiny steps (1,000 steps).
- Deterministic paths are smooth. To trace a smooth curve, you can take huge steps (20 steps).

Now, to skip steps, we must stop thinking of diffusion as a discrete sequence (($1 \dots 1000$)) and start treating it as a continuous curve in time $t \in [0, 1]$.

### **The Continuous Forward SDE**

If we take the limit as the number of steps $T \to \infty$, our discrete update rule becomes a **Stochastic Differential Equation (SDE)**.

Recall the discrete step:
$$
x_{t+1} = \sqrt{1-\beta} x_t + \sqrt{\beta} \epsilon
$$
In continuous time, this becomes the **Variance Preserving (VP) SDE**:
$$
d\mathbf{x} = \underbrace{-\frac{1}{2}\beta(t)\mathbf{x} \, dt}_{\text{Drift (Pull to 0)}} + \underbrace{\sqrt{\beta(t)} \, d\mathbf{w}}_{\text{Diffusion (Add Noise)}}
$$
- $dx$:The change in the image.
- $\beta(t)$: The continuous noise schedule.
- $d\mathbf{w}$: A standard Brownian motion (Wiener process).

---

### **The Score Function ($\nabla \log p$)**

The "magic" of diffusion is based on a quantity called the **Score Function**: the gradient of the log-density of the data.
$$
s(\mathbf{x}, t) = \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$
- *Intuition:* This vector points towards the "high probability" regions (clean images).
- **Crucial Link:** Our neural network  is actually learning this Score Function!
    $$
    \nabla_\mathbf{x} \log p_t(\mathbf{x}) \approx -\frac{\epsilon_\theta(\mathbf{x}, t)}{\sigma(t)} = -\frac{\epsilon_\theta(\mathbf{x}, t)}{\sqrt{1-\bar{\alpha}_t}}
    $$
- Predicting noise ($\epsilon$) is mathematically identical to calculating the gradient of the data density ($\nabla \log p$).

---

### **The Probability Flow ODE**

Song et al. (2021) proved that for any SDE of the form $dx = f(x,t)dt + g(t)dw$, there exists an **ODE** that shares the *exact same marginal probability densities* $p_t(x)$.

The general formula for this ODE is:
$$
d\mathbf{x} = \left[ \underbrace{f(\mathbf{x}, t)}_{\text{Drift}} - \frac{1}{2} \underbrace{g(t)^2}_{\text{Diffusion}} \underbrace{\nabla_\mathbf{x} \log p_t(\mathbf{x})}_{\text{Score}} \right] dt
$$

Now, let's **tighten the math** by plugging in our Diffusion variables.
1. **Substitute Drift :** $f(x,t)$: $-\frac{1}{2}\beta(t)\mathbf{x}$
2. **Substitute Diffusion :** $g(t)$: $\sqrt{\beta(t)}$
3. **Substitute Score :** $\nabla \log p$: $-\frac{\epsilon_\theta(\mathbf{x}, t)}{\sqrt{1-\bar{\alpha}_t}}$
$$
d\mathbf{x} = \left[ -\frac{1}{2}\beta(t)\mathbf{x} - \frac{1}{2} \beta(t) \left( -\frac{\epsilon_\theta(\mathbf{x}, t)}{\sqrt{1-\bar{\alpha}_t}} \right) \right] dt
$$

Simplifying gives us the **Final ODE Equation** that solvers like DPM-Solver use:
$$
\boxed{ \frac{d\mathbf{x}}{dt} = -\frac{1}{2}\beta(t) \left[ \mathbf{x} - \frac{\epsilon_\theta(\mathbf{x}, t)}{\sqrt{1-\bar{\alpha}_t}} \right] }
$$

#### **Interpretation of the ODE**

Look at the boxed equation. It describes the velocity of the image as we move from Noise ($t=T$) to Data ($t=0$).

* **The Term $\frac{\epsilon_\theta}{\sqrt{1-\bar{\alpha}}}$:** This is the scaled noise prediction.
* **The Term $(x - \text{Noise})$:** This estimates the "Clean Image" .
* **The Equation says:** "At every moment, move the image $x$ towards the estimated clean image $x_0$, scaled by the noise rate $\beta(t)$."

---

### **How ODE Solvers Skip Steps**

Once we define the process as an ODE:
$$
\frac{dx}{dt} = f(x, t)
$$
Our goal is to find $x(0)$ given $x(T)$. This is exactly what ODE Solvers do.

1. **Euler Method (DDPM / Ancestral)** is the naive approach.
    - Logic: "I am at $t=1000$. The slope points that way. I will take a tiny step."
    - Math: $x_{t-1} = x_t - \text{slope} \times \Delta t$.
    - Problem: It assumes the curve is a straight line. Since the diffusion curve is curved, taking a big step creates huge "Truncation Error." You fall off the curve.
    - Result: You assume you need $\Delta t = 0.001$ (1,000 steps).
2. Method 2: Higher-Order Solvers (Heun, Runge-Kutta, DPM-Solver) are "smart" solvers.
    - Logic: "I am at $t=1000$. Let me look at the slope here, AND estimate the slope at $t=950$, AND check the curvature."
    - Mechanism (Taylor Expansion): They use derivatives of the gradient to predict how the curve bends.
    - Result: They can accurately predict where the curve will be in a massive jump (e.g., $t=1000 \to t=950$).
    - Impact: We can reduce inference from 1,000 steps to 10-25 steps with almost no loss in quality.

More formally, since we now have a function $\frac{dx}{dt} = \Phi(x, t)$, we can use numerical integration.

Instead of taking 1,000 tiny Euler steps:
$$
x_{t-\Delta t} \approx x_t - \Phi(x_t, t)\Delta t
$$

We use a **Runge-Kutta** solver that looks ahead:
$$
\begin{align}
k_1 &= \Phi(x_t, t) \\
k_2 &= \Phi(x_t + \frac{h}{2}k_1, t + \frac{h}{2}) \\
x_{t-h} &= x_t - h \cdot k_2 + O(h^3)
\end{align}
$$

Because the error term is cubic $O(h^3)$ instead of linear $O(h)$, we can make the step size $h$ **massive** (skipping 50 steps at a time) while keeping the error low.

### Why use SDE Solvers? (Stochastic Differential Equations)

If ODEs are faster (20 steps), why do we sometimes still use SDEs (adding noise during inference)?

1. Error Correction: The ODE path is a tightrope.
    - If the neural network makes a slight error in prediction at step $t=900$, the ODE solver essentially "falls off the tightrope." It drifts into a weird region of latent space and creates artifacts.
    - SDEs are forgiving. By adding random noise at each step ($+ \sigma z$), we effectively "shake" the state back into the high-probability manifold. It corrects errors.
2. Quality & Texture (The "Blur" Problem)
    - ODE: Because it follows the *mean* path, it tends to result in slightly "averaged" or "conservative" images.
    - SDE: The added noise injects high-frequency details (grain/texture). SDE-generated images often look sharper and more realistic, even if they take longer to generate.

----

## **Text Guided Generation**

We use a backbone pre-trained language model to get text embedding `(batch_size, seq_len, dim_text)`

### **U-Net: Cross-Attention**
We use **cross-attention** layers, where **Query** = image features, and **Key/Value** is the text features;

Let's look at a specific layer inside the U-Net (e.g., the middle block).
1. Image Input (Query Source):
    - The U-Net is processing the noisy image feature map.
    - Shape: [B, Channels, H, W] (e.g., [1, 1280, 16, 16]).
    - Flatten: We flatten the spatial dimensions to make it a sequence.
    - $Q$ (Image): [B, 256, 1280] (where $256 = 16 \times 16$ pixels).
2. Text Input (Key/Value Source):
    - The frozen text embeddings from CLIP.
    - $K, V$ (Text): [B, 77, 768] (where $77$ is token count).
3. The Interaction:
    - We project $Q$ to dimension $d_{head}$ and $K, V$ to dimension $d_{head}$.
    - Attention Map ($Q \times K^T$):
        - Math: [B, 256, d] @ [B, d, 77] $\to$ [B, 256, 77].
        - Meaning: For every single pixel (1 of 256), the model calculates how relevant every single word (1 of 77) is.
        - Example: The pixel corresponding to the cat's eye will attend heavily to the word "cat" and "cyberpunk."
4. Output:
    - We multiply the map by $V$ (Text).
    - Result: [B, 256, 1280].
    - Reshape: Back to image [B, 1280, 16, 16].
**Summary**: The image pixels "query" the text prompt to retrieve relevant details.

### **DiT**

Used in Sora, Stable Diffusion 3. Transformers offer more flexible ways to inject conditioning.

#### Method 1: Adaptive Layer Norm (adaLN)

This is used for **Global Conditioning** (e.g., Time $t$, Class Label, or a Pooled Text Vector).
1. Input:
    - Noisy Image Patches (Main Stream): [B, N_patches, D_model].
    - Conditioning Vector $c$: A single vector [B, D_cond] (e.g., Time embedding + Pooled Text embedding).
2. Mechanism:
    - We do not just add $c$. We use an MLP (Regressor) to predict the Scale ($\gamma$) and Shift ($\beta$) for the LayerNorm.
    - $MLP(c) \to [\gamma, \beta]$.
    - $\text{Norm}(x) = \gamma(c) \cdot \frac{x - \mu}{\sigma} + \beta(c)$.
3. Effect: The text/time strictly controls the "gain" and "bias" of the entire network. If the prompt is "Night," the $\beta$ shift might darken all activations globally.

#### Method 2: Cross-Attention (Hybrid)

Similar to U-Net. We can interleave "Self-Attention" blocks (Image-to-Image) with "Cross-Attention" blocks (Image-to-Text) inside the Transformer.
- Stable Diffusion 3 uses this. It has a "MM-DiT" (Multimodal DiT) where Image and Text tokens pass through their own transformer streams but exchange information via attention.

#### Method 3: Token Appending (Pure Transformer)

This is the simplest, "ChatGPT-like" approach.
0. Concept: Treat Text and Image as just "tokens" in a long sequence.
1. Inputs:
    - Image Patches: $N_{img}$ tokens (e.g., 256 tokens).
    - Text Tokens: $N_{txt}$ tokens (e.g., 77 tokens).
2. Operation:
    - Get image embedding from certain encoder, and text embedding from the `lm_head`.
    - Simply Concatenate them: Sequence Length $= 256 + 77 = 333$.
    - Feed the whole [B, 333, D_model] tensor into a standard Transformer.
3. Self-Attention:
    - The Attention matrix is [333, 333].
    - Image tokens can attend to Text tokens (and vice versa) naturally.


### **Discrete Autoregressive Transformers (VQ-Modeling)**

Used by: DALL-E 1, VQ-GAN, Parti, MUSE, VQ-Diffusion.

- Tokenizer (VQ-VAE): train a separate model (ViT + Quantizer) to compress the image into a grid of Integers (e.g., [45, 992, 101, ...]).
- Vocabulary: a "Codebook" (e.g., 8192 possible image tokens).
- Embedding: Now we can use a standard Embedding Table, just like in NLP. Token #45 looks up vector #45.
- Generation: You generally use an Autoregressive Transformer (like GPT) to predict the next token, or a Masked Transformer (like BERT) to fill in missing tokens.

---
### **Classifier-Free Guidance (CFG)**

This is the most important trick in modern generative AI. It is the "magic dial" that forces the model to actually listen to your prompt.

#### **The Problem**

If you just train $p(x|c)$, the model often ignores the text. It might generate a generic high-quality image that vaguely matches the text but lacks specific details. It prioritizes "looking real" over "matching the prompt."

#### **The Solution**

We train a single model to do **two** things at once:
1. Conditioned: $\epsilon_\theta(x_t, t, c)$ (Predict noise given text).
2. Unconditioned: $\epsilon_\theta(x_t, t, \emptyset)$ (Predict generic noise).
    - Note: During training, we randomly replace the text $c$ with an empty string $\emptyset$ 10-20% of the time (Dropout) to teach the model both tasks.

#### **Inference (The Extrapolation)**

During sampling, we calculate the final noise prediction $\tilde{\epsilon}$ by **extrapolating** the difference between the two predictions:
$$
\tilde{\epsilon} = \underbrace{\epsilon_\theta(x_t, t, \emptyset)}_{\text{Generic Concept}} + w \cdot \underbrace{(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))}_{\text{The "Textness" Vector}}
$$
- $w$ (Guidance Scale): A scalar (usually $7.0 \sim 10.0$).
- Intuition: We calculate the direction that moves the image away from "generic genericness" and towards "your prompt." Then we push hard (multiply by $w$) in that direction.
- Result: High $w$ = Strict adherence to prompt (but can burn/fry the image). Low $w$ = Creative/ignored prompt.


### **Latent Diffusion Models (LDM / Stable Diffusion)**

Standard diffusion (like DDPM) operates in Pixel Space.
- Input: $512 \times 512 \times 3$ image.
- Cost: Extremely expensive. Every step processes millions of pixels.

Latent Diffusion (Rombach et al., 2022) moves the process to Latent Space.

#### **The Architecture**

1. VAE Compression: Train a standard VAE (Encoder $\mathcal{E}$, Decoder $\mathcal{D}$) to compress an image $x$ into a latent code $z = \mathcal{E}(x)$.
    - Example: $512 \times 512 \times 3 \to 64 \times 64 \times 4$. (Factor of 48x compression).
2. Diffusion in Latent Space: Train the Diffusion Model (U-Net/DiT) to generate $z_t$, not $x_t$.
    - The "Image" the diffusion model sees is essentially a dense feature map.
3. Decoding: Once the Diffusion loop finishes and outputs a clean latent $z_0$, run it through the VAE Decoder $x = \mathcal{D}(z_0)$.

Why? Efficiency. Running 50 steps on a $64 \times 64$ tensor is feasible on consumer GPUs. Running on $512 \times 512$ is not.

#### **Why using VAE**

If the Diffusion Model is doing the heavy lifting of generation, why do we care if the compressor is variational (probabilistic) or just a standard deterministic Autoencoder?

The short answer: We need the regularization (KL Divergence), not necessarily the stochastic sampling.

Diffusion models start from Pure Gaussian Noise 2$\mathcal{N}(0, I)$ and try to denoise it into a valid latent $z$.
The VAE adds the KL Divergence term to the loss:
$$
\mathcal{L} = \text{Recon} + \lambda \cdot D_{KL}(q(z|x) \| \mathcal{N}(0, I))
$$
This forces the encoder to pack the latent codes into a neat, standard unit sphere centered at 0.
- Result: The latent space $z$ "looks like" Gaussian noise.
- Benefit for Diffusion: This minimizes the domain gap. The Diffusion process adds Gaussian noise to something that is already quasi-Gaussian. The transition is smooth.

During Diffusion Training, we usually don't need the stochasticity $z \sim \mathcal{N}(\mu, \sigma)$.
In the original High-Resolution Image Synthesis with Latent Diffusion Models paper (Rombach et al.), they actually tested two types of regularization:
- KL-Reg (VAE): Standard VAE.
- VQ-Reg (Vector Quantized): Discrete codes (used in VQ-GAN).

Deterministic Trick: When training the Diffusion model, we often just take the Mean of the posterior $\mu = E(x)$ (or sample with very low variance) to get the latent $z_0$. We don't necessarily need the randomness there, because the Diffusion process itself adds massive amounts of noise ($x_t$).

So why VAE? It's not about the randomness; it's about the Geometry of the latent space. VAEs produce a smooth, continuous, and centered manifold that is easy for a Diffusion model to traverse.

---

### **SOTA Video Generation (Sora & The Future)**

How do we move from Images (2D) to Video (3D: Time + Height + Width)?

**The Old Way (2.5D U-Nets)**:Take a standard Image U-Net and add "Temporal Attention" layers that look across frames. This works for short clips but lacks global coherence.

**The New Way (Sora / DiT-based)**: Spacetime Patches Models like Sora treat video as a purely Sequence Modeling problem, utilizing the scalability of the DiT.

**Workflow**
1. Spacetime Latent: Compress the video (pixels) into a latent volume using a Video VAE.
    - Input (Raw Video): $(T_{frames}, H, W, C)$, e.g. 2 seconds at 30fps, $256 \times 256$ resolution.
        - Shape: (Batch=1, Channels=3, Frames=60, Height=256, Width=256)
    - 3D VAE Compression:
        - Space Compression: $8\times$ (Standard SD).
        - Time Compression: $4\times$ (Compresses 4 frames into 1 latent time-step).
        - Latent Channels: $4$ (Standard).
    - Output (Latent Volume):
        - Frames: $60 / 4 = 15$.
        - Height: $256 / 8 = 32$.
        - Width: $256 / 8 = 32$.
        - Latent Shape: (1, 4, 15, 32, 32)
2. 3D Patching:
    - Instead of cutting 2D squares from an image, we cut 3D Cubes (Spacetime Patches) from the video volume.
    - We define a Patch Size for the 3 dimensions $(t, h, w)$.
        - Patch dimensions: $(t_p=1, h_p=2, w_p=2)$.
        - Note: Usually $t_p=1$ or $2$ in latent space. If $t_p=1$, it grabs 1 latent frame (which represents 4 real frames).
        - Let's use $(1, 2, 2)$ for this example.
        - We slide this $1 \times 2 \times 2$ cutter over the $15 \times 32 \times 32$ volume.
    - A patch represents "This region of pixels for these 5 frames."
    - Tokens
        - Temporal Tokens ($N_t$): $\frac{\text{Latent Frames}}{\text{Patch Time}} = \frac{15}{1} = 15$
        - Height Tokens ($N_h$): $\frac{\text{Latent Height}}{\text{Patch Height}} = \frac{32}{2} = 16$
        - Width Tokens ($N_w$): $\frac{\text{Latent Width}}{\text{Patch Width}} = \frac{32}{2} = 16$
        - Total Tokens: $15 \times 16 \times 16 = \mathbf{3,840 \text{ tokens}}$.
3. Linearization: Flatten these cubes into a massive long sequence of tokens.
    - Input Sequence: `[Batch, 3840, Token_Dim]`
    - Each token represents a volume of: $Channels=4$, $Time=1$, $Height=2$, $Width=2$, with Raw Data Size: $4 \times 1 \times 2 \times 2 = 16$ values.
    - Projection: We map these 16 values to the Transformer Dimension $D_{model}$ (e.g., 1024) using a Linear Layer.
        - Mapping token dim (16) to model dim (1024) to get `[Batch, 3840, Model_Dim]`
4. DiT Processing:
    - Feed the sequence into a standard DiT.
    - The Transformer's Self-Attention handles the physics. It learns that "The ball in Patch A (Frame 1) must move to Patch B (Frame 2)."
    - It doesn't "know" it's processing video; it just sees tokens relating to each other.
5. Scaling: Because DiT scales with compute (Scaling Laws), simply throwing more GPUs and data at this architecture results in realistic physics, consistency, and object permanence.

Sora is a game changer since it proves that if you treat video generation as "Denoising 3D Noise" using a Transformer, you automatically get physical simulation properties emerging from the data, without explicitly programming physics engines.




---

# **Image v.s. Text Generation, Auto-Regressive v.s. Diffusion**

**WIP**
This is a sophisticated question that cuts to the core of why these two architectures look so different.
You are correct: The fundamental difference lies in the Conditional Distribution we assume ($p(y|\psi)$) and the nature of the "Latent" space.
Here is the breakdown of why Text is "Auto-Regressive Categorical" while Images are "Variational Gaussian," and how the "Variational" math connects them.



## 1. **The Tale of Two Distributions**
The architecture choices follow directly from the mathematical nature of the data.
- Text Generation (Discrete & Sequential)
    - The Data: Words are distinct, symbolic integers. There is no "halfway" between word ID 100 ("Cat") and word ID 101 ("Table").
    - The Assumption: Categorical Distribution.
    - The Process (Auto-Regressive): Because the data is sequential ($t_1 \to t_2 \to t_3$), we model the conditional probability of the next token.$p(x) = \prod p(x_t \mid x_{<t})$
    - The "Latent": There is no explicit "latent variable" in standard GPT training. The "embedding" is just a deterministic lookup table. The randomness only happens at the very end (Sampling).
- Image Generation (Continuous & Spatial)
    - The Data: Pixels are continuous values ($0.0 \to 1.0$). They have spatial correlation (neighbors look similar).
    - The Assumption: Gaussian Distribution.
    - The Process (Diffusion): We cannot model pixels sequentially (left-to-right) efficiently because images are 2D holistic structures. Instead, we model a sequence of noise levels.
    - The "Latent": The image itself is the latent variable, but it evolves over time steps $t$.

## **2. Deconstructing Diffusion**:
What are we assuming?
In your example (Text-Guided Image Editing/Generation), the workflow is distinct.
Let's look at Latent Diffusion (e.g., Stable Diffusion), which is the standard today.
- The Inputs & OutputsInput ($u$):
    - Text: Embedded via CLIP/T5 (Frozen).
    - Image State ($x_t$): This is the crucial part. We don't input "pixels." We input a Noisy Latent Tensor.
        - Note: If using Stable Diffusion, we first use a VAE Encoder to compress the pixels into a latent space
            - $z_0$, then add noise to get
            - $z_t$.
- Model ($f_\theta$): A U-Net or DiT (Diffusion Transformer).
    - It takes Noisy Latent $z_t$ + Text Embedding.
    - It outputs $\psi = \hat{\epsilon}$ (Predicted Noise).
- Assumed Distribution: Conditional Gaussian.
    - The model assumes the "cleaner" image $z_{t-1}$ is drawn from a Gaussian centered around the current state adjusted by the predicted noise.
    - $p(z_{t-1} \mid z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \sigma_t^2 I)$
- Inference:
    - We sample noise $z_T \sim \mathcal{N}(0, I)$.
    - We iteratively remove noise using the predicted $\mu_\theta$.

## **3. The "Variational" Mystery: What are we varying?**

You asked: "Why here we use a 'variational' way? What are we essentially varying?"
This terminology comes from Variational Inference, a branch of Bayesian statistics.
- The Problem: We want to maximize the likelihood of real images $p(x)$. But calculating $p(x)$ directly is impossible because we'd have to integrate over all possible ways an image could be formed.
- The Variational Solution: Instead of calculating $p(x)$ exactly, we optimize a Lower Bound (the ELBO). To do this, we introduce an approximate posterior distribution $q$.
- What are we varying? We are varying the path of the image from Noise to Data.
    - In a VAE, we vary the single latent vector $z$.
    - In Diffusion, we vary the entire trajectory of states $x_T \to x_{T-1} \dots \to x_0$.
**Diffusion models are mathematically just Hierarchical VAEs with infinitely many layers.**
- VAE: 1 layer of latent variables ($z$).
- Diffusion: 1000 layers of latent variables ($x_{999}, x_{998} \dots$).
- We treat every step of adding/removing noise as a single layer in a massive VAE. The "Variational Lower Bound" math simplifies beautifully into the MSE Loss you use in code: $\|\epsilon - \epsilon_\theta\|^2$.

## **4. "Latent" vs. "Noise": How they come into play**
This is often the most confusing part. Let's separate them:
- A. The "Latent" (The Canvas)
    - In VAEs/Latent Diffusion: The "Latent" $z$ is the compressed representation of the image.
    - Role: It is the State. It is the thing being transformed.
    - Analogy: It is the block of marble. At step $T$, it is a rough block. At step 0, it is a statue.
- B. The "Noise" (The Tool)
    - In Diffusion: $\epsilon \sim \mathcal{N}(0, I)$.
    - Role: It is the Source of Stochasticity.
        - Training: We add noise to destroy the image (Forward Process). The model learns to predict this specific noise instance.
        - Inference: We sample noise to kickstart the generation.
    - Analogy: It is the chisel strike. We need randomness to generate new statues; otherwise, we'd carve the exact same statue every time.



# Part of the Prompts Used for this note

- For image embedding models (like ViT), or in general for all embedding models (like using BERT for text embeddings, or even LLM for word embeddings), are we essentially using the same ideas for Auto-Encoders: that all data lie in the smaller manifold compared to the ambient space? Like a compression, we enforce the representation, our 'encoded' way of input from the original, raw input, will be better suited for down-stream tasks, like adding a LM head or MLP for certain classification work, where we then assume a certain (categorical) distribution?

- Following this question, even we name 'intermediate representation' as part of the whole model, can we effectively assume encoder and decoders are decoupled? Where the embedding of the encoder can be used, or easily fine-tuned, for other down-stream tasks? Does this apply to AE, or VAE's encoder and decoder as well?

- Can we unify the way VAE is doing encoding, with how we form our 'stochastic' part of the model? i.e. we are outputing $\psi$ with local parameters for the distribution, whereas in VAE for the latent variables we are essentially outputting mean and var for a guassian distribution? When we say our latent does not follow a specific distribution, we are saying the local parameters themselves?

- In VAE, we say given the input, we have the latent from the encoder, where it is local parameters of a distribution; can we apply this to ALL the reprensentation / embeddings for other tasks, e.g. language modeling? Instead of output a fixed vector for a word, why don't we follow the latent idea of VAE and train the encoder in a way that it output a parameters of the distribution? Adding the stochastic even on the fixed embedding vectors.  

- Taking a step back, a more fundamental question is, what is the purpose of VAE? In image understanding, we have both text and image input, where the image going to ViT for an embedding, and we also have text embedding from our decoder's embedding table, and then we do certain cross-attention and the decoding part will be 'controlled', (and for image generation as well). For VAE, however, we input an image, and we are output an image as well, there's no explicit control in it, is it? The best we can do is outputting the exact same image? Why we need VAE, or in general, this task at all? How do we add 'control' in the images we are generating?

- When we say why probabilistic embedding is not the default, we say usually we use the mean only. So, with this interpretation, essentially the output embedding vector is still not a point mass, but a distribution parameter? Isn't this contradict to what we were saying? 2. We also say, this is for discriminative tasks. But for generation tasks including LLM itself, why we still don't think it is necessary? 3. Then why this is absolutely necessary for VAE? If ViT and LLaVA can handle 'holes' on the manifold well, why can't VAE? If we switch the decoder to a very powerful transformer based decoder (parameters in terms of hundreds of Billions or even trillion), like modern multi-modal understanding / generation models, then we don't necessarily use latent as both \mu and \sigma? More importantly, the question is why for VAE this is required, but why current VLM (like GLM or Gemma) doesn't use it?

- The fundamental difference between text generation and image generation, or, auto-regressive way and diffusion way, is the underlying conditional distribution we are assuming? For text generation, given a prompt, we embed it to be (seq_len, dim), and this embedding could be from previous step of the decoder (as we are doing auto-regressive way), then, given our decoder f_\theta, we output a logits and softmax over the whole vocabulary for the next most probable word (which we could do sampling based on top-k, top-p as we are assuing a categorical distribution for next word given prompts). On the other hand, for image generation, where we usually use diffusion way, given a text and a image pair (e.g. text: change the cat in this image to a dog, and image: a picture of a cat on a sofa), we embed the text using lm_head of a decoder, and we embed the image using the ViT, and then, given (seq_len, dim) and (patched / compressed pixels, dim), we do diffusion. Now, what conditional distribution are we assuming? Why here we use a 'variational' way? What are we essentially varying? How do 'latent' and 'noise' come into play?
