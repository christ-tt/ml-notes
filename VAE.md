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
$$\text{ELBO} = \mathbb{E}_{q} \left[ \log \frac{p(x, z)}{q(z \mid x)} \right]$$
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
- The Likelihood $p_\theta(u \mid z)$ is the Score. It measures the distance between the Arrow ($\hat{u}$) and the Target ($u$).$$\text{Log Likelihood} \propto - \| \text{Target}(u) - \text{Arrow}(\hat{u}) \|^2$$
So, $p_\theta(u \mid z)$ is the value we maximize. We do this by moving the Arrow ($\hat{u}$) closer to the Target ($u$). $\hat u = f_\theta(z), u \sim \mathcal{N}(\hat u, I)$, i.e. $p_\theta(u \mid z) = p(u \mid \psi) = \mathcal{N}(u; \hat u, \sigma^2I)$

**Goal**: Show that minimizing the KL divergence between the Approximate Posterior $q_\phi(z \mid u)$ and the True Posterior $p_\theta(z \mid u)$ is equivalent to maximizing the ELBO (Evidence Lower Bound).

$$\begin{align}
\mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p_\theta(z \mid u)\big) &= \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{q_\phi(z \mid u)}{p_\theta(z \mid u)} \right] \\
&= \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} [\log p_\theta(z \mid u)]
\end{align}$$
Apply Bayes' Rule to the true posterior: $p_\theta(z \mid u) = \frac{p_\theta(u \mid z) p(z)}{p_\theta(u)}$.
$$
\begin{align}
\dots &= \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} \left[ \log \frac{p_\theta(u \mid z) p(z)}{p_\theta(u)} \right] \\
&= \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} [\log p_\theta(u \mid z)] - \mathbb{E}_{z \sim q_\phi} [\log p(z)] + \underbrace{\mathbb{E}_{z \sim q_\phi} [\log p_\theta(u)]}_{\log p_\theta(u) \text{ is const w.r.t } z}
\end{align}
$$
Now, rearrange the terms to group the KL Divergence to Prior and the Reconstruction Loss:
$$\begin{align}
\mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p_\theta(z \mid u)\big) &= \underbrace{\left( \mathbb{E}_{z \sim q_\phi} [\log q_\phi(z \mid u)] - \mathbb{E}_{z \sim q_\phi} [\log p(z)] \right)}_{\mathrm{KL}(q_\phi(z \mid u) \,\|\, p(z))} - \mathbb{E}_{z \sim q_\phi} [\log p_\theta(u \mid z)] + \log p_\theta(u)
\end{align}$$
Thus, the relationship is:
$$\log p_\theta(u) = \underbrace{\mathbb{E}_{z \sim q_\phi} [\log p_\theta(u \mid z)] - \mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p(z)\big)}_{\text{ELBO}(\theta, \phi)} + \underbrace{\mathrm{KL}\big(q_\phi(z \mid u) \,\|\, p_\theta(z \mid u)\big)}_{\ge 0}$$

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
        $$\log p(x \mid z) \propto -\frac{1}{2} (x - D(z))^T C^{-1} (x - D(z)) - \frac{1}{2} \log |C|$$
        We could
            - Learn it (Heteroscedastic): The Decoder outputs both the mean $\mu_x$ and the variance $\sigma_x^2$ for every pixel.
                - Pros: The model learns which parts of the image are detailed (low variance) vs. noisy/texture (high variance).
                - Cons: Unstable training. It often collapses to predicting zero variance (infinite likelihood).
            - Fix it (Homoscedastic) \[**standard**\]
                - We simply assume $C = \sigma^2 I$ is a fixed scalar constant (hyperparameter) for all data points.
                - If $C = I$ (identity matrix), the term $\log |C|$ becomes constant and vanishes from the optimization.
                - The likelihood term simplifies purely to:$$\mathcal{L}_{\text{recon}} \propto - \| x - D(z) \|^2$$
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



# **Hierarchical VAE**
To understand Diffusion, we don't need to learn a "new" algorithm; we just need to take a **Hierarchical VAE (HVAE)** and take it to its logical extreme.

An HVAE is simply a VAE that doesn't stop at one layer of latent variables. Instead of a single compression step, it performs a sequence of them.

**Motivation**: Trying to compress a complex $256 \times 256$ image into a single vector $z$ causes "information bottleneck." The model is forced to average out fine details (high frequency) to save the global structure (low frequency).

## **Architecture**
### **A. The Encoder (Bottom-Up / Inference)**

We compress the data step-by-step.

* **Step 1:** Map Image $x$ to $z_1$ (Low-level features, like edges).
* **Step 2:** Map $z_1$ to $z_2$ (Mid-level shapes).
* **Step T:** Map $z_{T-1}$ to $z_T$  (High-level semantics, "Cat").
$$q(z_{1:T} \mid x) = q(z_1 \mid x) \times q(z_2 \mid z_1) \times \dots \times q(z_T \mid z_{T-1})$$
We are assuming a **Markov Chain**: $z_{t}$ only depends on $z_{t-1}$ for $t > 1$.

### **B. The Decoder (Top-Down / Generation)**

We reconstruct the image step-by-step.

* **Step 1:** Sample abstract concept $z_T$.
* **Step 2:** Flash out details to get $z_{T-1}$.
* **Final Step:** Generate pixels $x$ from $z_1$.
$$p(x, z_{1:T}) = p(z_T) \times p(z_{T-1} \mid z_T) \times \dots \times p(x \mid z_1)$$

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

### **Diffusion Trick - Inverting Encoder**
To make the math clean (and to get to Diffusion Models), we perform a slight trick. We rewrite the ELBO using Bayes' rule on the Posterior side.

Instead of defining $q(z_t \mid z_{t-1})$ (Bottom-Up), let's imagine we could define the Reverse Posterior $q(z_{t-1} \mid z_t, x)$.
- Idea: "Given that I'm at the noisy state $z_t$, and I know the original image $x$, what was the previous step $z_{t-1}$?"

**Note**: The ELBO Term:
$$
\log \frac{p(z_{t-1} \mid z_t)}{q(z_t \mid z_{t-1})}
$$
We **can't** simply negate this to form a KL divergence:
$$
- \log \frac{q(z_t \mid z_{t-1})}{p(z_{t-1} \mid z_t)},
$$
which is NOT a valid KL Divergence. KL Divergence $D_{KL}(A \| B)$ requires $A$ and $B$ to be distributions over the same variable.Here, $q$ gives probabilities for $z_t$, but $p$ gives probabilities for $z_{t-1}$.

Thus, we **invert** the encoder. To compare the Encoder to the Decoder, we must force the Encoder to *look "backward"* (from $t$ to $t-1$), just like the Decoder. We want to convert the forward rule $q(z_t \mid z_{t-1})$ into a reverse posterior $q(z_{t-1} \mid z_t)$.

Using Bayes' Rule:
$$q(z_{t-1} \mid z_t) = \frac{q(z_t \mid z_{t-1}) \, q(z_{t-1})}{q(z_t)}$$

If we rewrite the ELBO using this formulation (which is standard in Diffusion math), the sum becomes a beautiful chain of KL divergences:
$$\text{ELBO} = \underbrace{\mathbb{E}[\log p(x \mid z_1)]}_{\text{Reconstruction}} - \sum_{t=2}^T \underbrace{D_{KL}\Big( q(z_{t-1} \mid z_t, x) \,\|\, p(z_{t-1} \mid z_t) \Big)}_{\text{Denoising Matching}} - \underbrace{D_{KL}(q(z_T \mid x) \| p(z_T))}_{\text{Prior Matching}}$$

### **The Mathematical Bridge to Diffusion**

This is the critical realization that leads to Diffusion Models.

Imagine a specific type of HVAE with **three constraints**:

1. **Infinite Depth:** We add more and more layers ($T\to \infty$).
2. **Same Dimension:** Every latent layer $z_t$  has the **same shape** as the image $x$ (no compression in size, only in information).
3. **Fixed Encoder:** This is the kicker.
* In HVAE, we *learn* the Encoder $q_\phi$.
* In this special case, we **fix** the Encoder to be a simple, non-learnable noise injector.
* $q(z_t \mid z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta} z_{t-1}, \beta I)$. (Just adding Gaussian noise).

**If you do this, the HVAE becomes a Diffusion Model.**

#### **Comparison Table**

| Feature | **Standard HVAE** | **Diffusion Model (VDM)** |
| --- | --- | --- |
| **Latent Layers** | Several ($T \approx 5 \sim 10$) | Many ($T \approx 1000$) |
| **Latent Dim** | Getting smaller (Compressed) | Same size as Image (Full Res) |
| **Encoder $q(z \mid x)$** | **Learned** Neural Network | A fixed Linear Schedule (solver), not trainable
| **Decoder $p(x \mid z)$** | Learned Neural Network | A Neural Network (U-Net)
| **Latent Meaning** | Abstract Features (Edges, Shapes) | Noisy Images (Pixel soup) |

## **VAE to Diffusion: The Mental Shift**

* **HVAE:** "I will learn a smart hierarchy of features to compress the image."
* **Diffusion:** "I will define the 'encoding' simply as destroying the image with noise layer-by-layer. Then, I will treat the 'decoding' as a massive HVAE that learns to reverse this destruction."

So, mathematically, a Diffusion model **is** an HVAE where the inference path is fixed to be a noise process, and we only train the generative path to undo it.


## Questions

- For image embedding models (like ViT), or in general for all embedding models (like using BERT for text embeddings, or even LLM for word embeddings), are we essentially using the same ideas for Auto-Encoders: that all data lie in the smaller manifold compared to the ambient space? Like a compression, we enforce the representation, our 'encoded' way of input from the original, raw input, will be better suited for down-stream tasks, like adding a LM head or MLP for certain classification work, where we then assume a certain (categorical) distribution?

- Following this question, even we name 'intermediate representation' as part of the whole model, can we effectively assume encoder and decoders are decoupled? Where the embedding of the encoder can be used, or easily fine-tuned, for other down-stream tasks? Does this apply to AE, or VAE's encoder and decoder as well?

- Can we unify the way VAE is doing encoding, with how we form our 'stochastic' part of the model? i.e. we are outputing $\psi$ with local parameters for the distribution, whereas in VAE for the latent variables we are essentially outputting mean and var for a guassian distribution? When we say our latent does not follow a specific distribution, we are saying the local parameters themselves? 

- In VAE, we say given the input, we have the latent from the encoder, where it is local parameters of a distribution; can we apply this to ALL the reprensentation / embeddings for other tasks, e.g. language modeling? Instead of output a fixed vector for a word, why don't we follow the latent idea of VAE and train the encoder in a way that it output a parameters of the distribution? Adding the stochastic even on the fixed embedding vectors.  

- Taking a step back, a more fundamental question is, what is the purpose of VAE? In image understanding, we have both text and image input, where the image going to ViT for an embedding, and we also have text embedding from our decoder's embedding table, and then we do certain cross-attention and the decoding part will be 'controlled', (and for image generation as well). For VAE, however, we input an image, and we are output an image as well, there's no explicit control in it, is it? The best we can do is outputting the exact same image? Why we need VAE, or in general, this task at all? How do we add 'control' in the images we are generating?

- When we say why probabilistic embedding is not the default, we say usually we use the mean only. So, with this interpretation, essentially the output embedding vector is still not a point mass, but a distribution parameter? Isn't this contradict to what we were saying? 2. We also say, this is for discriminative tasks. But for generation tasks including LLM itself, why we still don't think it is necessary? 3. Then why this is absolutely necessary for VAE? If ViT and LLaVA can handle 'holes' on the manifold well, why can't VAE? If we switch the decoder to a very powerful transformer based decoder (parameters in terms of hundreds of Billions or even trillion), like modern multi-modal understanding / generation models, then we don't necessarily use latent as both \mu and \sigma? More importantly, the question is why for VAE this is required, but why current VLM (like GLM or Gemma) doesn't use it?

- The fundamental difference between text generation and image generation, or, auto-regressive way and diffusion way, is the underlying conditional distribution we are assuming? For text generation, given a prompt, we embed it to be (seq_len, dim), and this embedding could be from previous step of the decoder (as we are doing auto-regressive way), then, given our decoder f_\theta, we output a logits and softmax over the whole vocabulary for the next most probable word (which we could do sampling based on top-k, top-p as we are assuing a categorical distribution for next word given prompts). On the other hand, for image generation, where we usually use diffusion way, given a text and a image pair (e.g. text: change the cat in this image to a dog, and image: a picture of a cat on a sofa), we embed the text using lm_head of a decoder, and we embed the image using the ViT, and then, given (seq_len, dim) and (patched / compressed pixels, dim), we do diffusion. Now, what conditional distribution are we assuming? Why here we use a 'variational' way? What are we essentially varying? How do 'latent' and 'noise' come into play?

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



