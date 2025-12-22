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



## **Recap: Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)**

### **Frequentist viewpoint and (Log) likelihood**
From the **frequentist** perspective, probability is defined via frequencies:
- For discrete variables, probabilities are estimated by counting occurrences
- For continuous variables, probability density is inferred from samples

Suppose we observe a dataset: $D= \{x_1, x_2, \dots, x_N\}$ and assume the data are i.i.d. given model parameters $\theta$ .
The **likelihood** of the dataset under the model is: $$P(D \mid \theta) = \prod_{k=1}^{N} P(x_k \mid \theta)$$
Directly working with products of probabilities is numerically unstable (values become extremely small). Therefore, we maximize the **log-likelihood** instead:
$$\log P(D \mid \theta) = \sum_{k=1}^{N} \log P(x_k \mid \theta)$$

Because the logarithm is monotonic, maximizing likelihood and maximizing log-likelihood are equivalent.

### Maximum Likelihood Estimation (MLE)
We assume a *family of distributions* parameterized by $\theta$ (e.g. Gaussian with mean and variance), and choose parameters that best explain the observed data: $$
\begin{align} 
\hat \theta_{\text{MLE}} &= \arg \max_\theta \log P(D\mid\theta) \\
&= \arg \max_\theta \sum_{k=1}^N \log P(x_i\mid\theta) \\
&= \arg \min_\theta \frac{1}{N}\sum_{k=1}^N -\log P(x_i\mid\theta) \\
&= \arg \min_\theta \mathbb E_{x \sim \hat p(x)} \left [- \log P(x\mid\theta) \right]
\end{align}$$ where $\hat p(x)$ is the empirical data distribution, i.e. sampling from $\hat p(x)$ means uniformly pick a data point from the dataset.
	Interpretation: the probability if I pick one observation uniformly at random from my **dataset**
	Given dataset $D = \{ x_1, x_2, \dots x_N\}$, the empirical distribution: $$\hat p(x) = \frac{1}{N} \sum_{k=1}^N \delta (x - x_k)$$ 
	where $\delta (\cdot)$ is the Dirac delta
		$\delta(x - a)$ is zero everywhere except $x = a$ ; 
		$\int_{-\infty}^{\infty} f(x)\,\delta(x-a)\,dx = f(a)$  

i.e. MLE fits model parameters so that the model is most likely to generate the observed samples.

### KL Divergence Equivalence of MLE
Let $p_{\text{data}}(x)$ be the true, unknown data distribution, and $p(x \mid \theta)$ be the model distribution, then $$\hat \theta_{\text{MLE}} = \arg \min_\theta \mathrm{KL}\big(p_{\text{data}}(x) \,\|\, p(x\mid \theta)\big)$$
where the expanded form $$\mathrm KL = \int p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p(x \mid \theta)} dx$$ 
We already established $$
\hat \theta_{\text{MLE}} = \arg \min_\theta \mathbb E_{x \sim \hat p(x)} \left [- \log P(x\mid\theta) \right] $$
By **Law of Large Numbers** , we have as $N \to \infty$ , $\hat p(x) \to p_{\text{data}}(x)$ .
So asymptotically, $\mathbb E_{x \sim \hat p(x)}[\cdot] \;\approx\; \mathbb E_{x \sim p_{\text{data}}(x)}[\cdot]$ 

Now, $$
\hat \theta_{\text{MLE}} = \arg \min_\theta \mathbb E_{x \sim p_{\text{data}}(x)} \left [- \log p(x\mid\theta) \right] $$
which is essentially the **cross-entropy** between $p(x \mid \theta)$ and $p_{\text{data}}(x)$.
Since the *entropy* of $p_{\text{data}}$ is **constant** w.r.t. $\theta$ , as the expectation only depends on the true distribution, not $\theta$, now, 
	Recall basic logarithm and linearity of expectation $$\begin{align*} \log (\frac{A}{B}) &= \log(A) - \log (B) \\ \mathbb E[-\log A] & = \mathbb E[\log B - \log A - \log B] \\ &= \mathbb E[\log\frac{B}{A}] - \mathbb E[\log B] \end{align*}$$
$$\begin{aligned} \mathbb E_{p_{\text{data}}}[-\log p(x\mid\theta)] &= \mathbb E_{p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(x)}{p(x\mid\theta)} \right] - \mathbb E_{p_{\text{data}}}[\log p_{\text{data}}(x)] \\ &= \mathrm{KL}\big(p_{\text{data}}(x) \,\|\, p(x\mid \theta)\big) + H(p_{\text{data}}) \end{aligned}$$

Now, with entropy of the data equals zero, we've established $$\hat \theta_{\text{MLE}} = \arg \min_\theta \mathbb E_{x \sim p_{\text{data}}(x)} \left [- \log p(x\mid\theta) \right] = \arg \min_\theta \mathrm{KL}\big(p_{\text{data}}(x) \,\|\, p(x\mid \theta)\big)$$
MLE is equivalent to minimizing the KL divergence from the true data distribution to the model distribution.
	It minimize *forward* KL, not symmetrically, potentially 
		* Penalized heavily for missing data modes
		* But not penalized much for placing mass where data doesn't exist 
	This asymmetry matters a lot later for AE vs VAE vs diffusion vs GANs.

### When MLE reduces to Mean Squared Error (MSE)
Assume the model: $$P(x \mid \theta) = \mathcal{N}(x \mid \mu_\theta, \sigma^2 I) $$
The negative log-likelihood for a single sample is, let $c$ be constant, $$- \log P(x\mid \theta) = c \frac{1}{2\sigma^2}\|x-\mu_\theta\|^2$$
Thus, $$\arg\max_\theta \log P(D \mid \theta) \;\;\Longleftrightarrow\;\; \arg\min_\theta \sum_{k=1}^N \|x_k - \mu_\theta\|^2$$
Minimizing mean squared error is equivalent to MLE under isotropic Gaussian noise assumption, which is also the reason why MSE appears ubiquitously in regression, autoencoders, and reconstruction losses.


### **When data is scarce: Maximum A Posteriori (MAP)**
When the dataset is small (e.g., flipping a coin only a few times), MLE can overfit.
In such cases, we introduce **prior knowledge** over parameters.

Using Bayes’ rule:

$$P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)} \;\;\propto\;\; P(D \mid \theta) P(\theta)$$
- $P(\theta)$: prior distribution (encodes beliefs or regularization)
- $P(D \mid \theta)$: likelihood
- $P(\theta \mid D)$: posterior distribution

MAP chooses the parameter that maximizes the posterior:$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta \log P(D \mid \theta) + \log P(\theta)$$
**Interpretation:**
- MLE: data-only fitting
- MAP: data + prior regularization

In practice:
- Gaussian prior → L2 regularization
- Laplace prior → L1 regularization





# Variational Auto Encoder
