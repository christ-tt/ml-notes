---
tags:
  - ML
  - Theory
  - Work
---

## Learning

We consider supervised or self-supervised learning settings where we observe samples $x \in \mathcal X$ drawn i.i.d. from an **unknown data-generating distribution** $p_{\text{data}}(x)$ where,
- $p_{\text{data}}(x)$ is **never known** and **never assumed to have a parametric form**.
- Learning proceeds only via a finite dataset $D = \{x_1, x_2, \dots, x_N\} \sim p_{\text{data}}$.
  

The goal of learning is to construct a **model** $f_\theta$ that **approximates** the data, by 
- Maximizing Likelihood: minimizing KL divergence;
- Minimizing Loss under the true data distribution.
$$
\theta^\star \approx \arg \min_\theta \mathbb E_{x \sim p_{\text{data}(x)}}\big [\mathcal L(x; \theta)\big]$$



## Likelihood

Given $\theta \in \Theta$ denote model parameters, we have 
$$\hat x_\theta = f_\theta(\cdot)$$
where:
- $\hat x_\theta$ is the model’s **point prediction** or **mean prediction**
- This is **deterministic**, and no randomness at this point.

To connect deterministic predictions to probabilistic learning, we introduce a **likelihood function**: given parameters $\theta$, the model defines a probability **distribution** over possible observations $x$.
$$p(x \mid \theta)$$

In practice, we introduce **noise** (observation model)
$$x = \hat x_\theta + \varepsilon$$

we implicitly factor
$$\theta \;\longrightarrow\; \hat x_\theta \;\longrightarrow\; x$$
from deterministic model output, to likelihood distribution.

We also have **reparameterization**: 
$$p(x \mid \theta) \;\equiv\; p(x \mid \hat x_\theta)$$
where
* $\hat x_\theta$ is the man / location parameter
* randomness comes from $\epsilon$, not from $\theta$, as $\theta$ determines a deterministic prediction $\hat x_\theta$,
- all randomness in x is modeled **conditionally on** $\hat x_\theta$.

This factorization is valid because, once we know the model's prediction $\hat x_\theta$, the parameters $\theta$ no longer matter for generating $x$:
$$x \;\perp\!\!\!\perp\; \theta \;\mid\; \hat x_\theta.$$


Our model, $f_\theta$ , is not trying to approximate $p_\text{data}$ directly. Instead, it approximates a **decision function** derived from the loss.
Given our learning objective $$\theta^* = \arg \min_\theta \mathbb E_{x \sim p_{\text{data}}} \big [\mathcal L(x;\theta)\big]$$ our optimal predictor $$f^* = \arg \min_f \mathbb E_{x \sim p_{\text{data}}} \big [\mathcal L(x, f(x))\big]$$


where the specific form of the loss $\mathcal L$ depends on how the data sample $x$ is structured and how the model is tasked to use it.
* **Supervised Learning**: 
	* $x = (\mu, y)$ where $\mu$ is the input and $y$ is the target; 
	* the loss measured predictive error $\mathcal L(y, f_\theta(\mu))$
* **Self-supervised Learning / Autoencoders**: 
	* $x$ is both input and target; 
	* the loss measures reconstruction fidelity $\mathcal L(x, \hat x_\theta)$. 

---

## Noise
**Noise model**, or **observation model** is defined to be the conditional distribution $$
p(x \mid \hat x_\theta)
$$It specifies how **deviations** between reality and the model prediction are treated.

  
Equivalently, we assume:
$$x = \hat x_\theta + \varepsilon, \quad \varepsilon \sim p_\varepsilon(\cdot)$$,

where:
- $\varepsilon$ is an abstract noise variable,
- $p_\varepsilon$ is chosen by the modeler.

Essentially,
- The noise model is not a claim about the true data distribution.
- It is a modeling choice that defines how prediction errors are penalized.


---

## Loss
- MSE Loss - Gaussian Noise, the *function* is the conditional *mean* $$\begin{align*} \mathcal L(y, \hat y) &= \|y - \hat y\|^2 \\ f^*(\mu) &= \mathbb E[y\mid \mu] \end{align*}$$
- MAE Loss - Laplace Noise, the *function* is the conditional *median* $$ \begin{align*} \mathcal L(y, \hat y) &= |y - \hat y| \\ f^*(\mu) &= \text{median}(y\mid \mu) \end{align*}$$
- Cross-Entropy Loss, the network approximates the *entire conditional distribution* $$ \begin{align*} \mathcal L(y, \hat y) &= -\log \hat p(y) \\ f^*(\mu) &= p_{\text{data}}(y \mid \mu) \end{align*}$$

i.e. our model approximates a target mapping that is implicitly defined by the data distribution and the learning objective.


By maximizing likelihood, we are essentially finding the parameters that make the observed noise most plausible.


---



## **5. Likelihood choice and induced loss**

Given a noise model, the learning objective is derived via the **negative log-likelihood**:

\mathcal L(x, \hat x_\theta) \;\stackrel{\text{def}}{=}\; -\log p(x \mid \hat x_\theta).

  

Thus:

  

> **A loss function is simply a negative log-likelihood under an assumed noise model.**

  

### **Examples**

|**Noise model** p(x \mid \hat x)|**Interpretation**|**Resulting loss**|
|---|---|---|
|\mathcal N(\hat x, \sigma^2 I)|Gaussian noise|Mean Squared Error|
|Laplace(\hat x, b)|Heavy-tailed noise|L1 loss|
|Bernoulli(\hat x)|Binary outcomes|Binary cross-entropy|
|Categorical(\hat x)|Multiclass outcomes|Softmax cross-entropy|

In each case:

- the model outputs parameters of a distribution (e.g., mean, logits),
    
- the likelihood converts that output into a probability of observing x.
    

---

## **6. Likelihood vs true data distribution**

  

It is essential to distinguish:

  

### **True data distribution**

  

p_{\text{data}}(x)

- Unknown
    
- Potentially complex, multimodal, irregular
    
- Never explicitly parameterized or assumed
    

  

### **Likelihood / noise model**

  

p(x \mid \hat x_\theta)

- Fully chosen by the modeler
    
- Defines error sensitivity and robustness
    
- Determines the geometry of the loss
    

  

Thus:
  
> **We do not assume a form for** p_{\text{data}}(x)**.**
> **We only assume a form for how model errors are measured.**

---

**Modeling** is **not** dictated by the data; we use the following for data-generation $$y = f_\theta(\mu) + \epsilon$$ where the only assumptions we make
- $f_\theta(\mu)$: systematic, predicable part
- $\epsilon$: everything we cannot or choose not to model
which is not two objectives, but a decomposition of responsibility.

Noise means how we choose to interpret discrepancies between model prediction and data; different assumptions of the distribution of the noise are how we *think* we should interpret the errors. 
* For example, normal means small errors are common, large errors are increasingly unlikely; errors are symmetric, and errors should be penalized quadratically, *L2*
* Laplace noise assume mostly clean data, but occasionally huge outliers *L1* 

**Unify output-fitting and noise**
From the model $$y = f_\theta(\mu) + \epsilon$$
we get the likelihood $$p(y\mid \mu, \theta) = \mathcal N(y \mid f_\theta(\mu), \sigma^2)$$answering: how plausible is the observed output given the model prediction?

Negative log-likehood is exactly MSE, and there's only a single objective $$-\log p(y \mid u, \theta) \propto \|y - f_\theta(u)\|^2$$ $$ \arg\min_\theta \sum_i \|y_i - f_\theta(u_i)\|^2$$
- Fits the outputs = make likelihood high
- Handle noise = define how deviations are penalized.

Given parameters $\theta$, the model produces either 
- a point prediction $\hat x_\theta=f_\theta(\cdot)$ (regression / AE) or 
- distribution parameters $\eta_\theta(\cdot)$ (classification / LLM). 
We then specify an *observation/noise* model $p(x\mid \hat x_\theta)$ (or $p(x\mid \eta_\theta)$), which is the only place where “noise” enters. 
Training chooses $\theta$ to maximize the probability of the observed dataset $D=\{x_i\}_{i=1}^N$_:_

$$\hat\theta=\arg\max_\theta \log p(D\mid \theta)=\arg\max_\theta \sum_{i=1}^N \log p(x_i\mid \theta) \equiv \arg\max_\theta \sum_{i=1}^N \log p(x_i\mid \hat x_{\theta,i}),$$

where $\hat x_{\theta,i}=f_\theta(\cdot)$ is the model prediction for sample i. Defining the per-sample loss as the **negative log-likelihood**,
$$
\mathcal L(x_i;\theta)\;\stackrel{\text{def}}{=}\;-\log p(x_i\mid \theta)=-\log p(x_i\mid \hat x_{\theta,i}),
$$
the same objective becomes empirical risk minimization:

$$\hat\theta=\arg\min_\theta \sum_{i=1}^N \mathcal L(x_i;\theta).$$

Thus there are not two separate objectives: 
- “fit outputs” is achieved by making $p(x_i\mid \hat x_{\theta,i})$ large (so $x_i$ is close to $\hat x_{\theta,i}$), 
- while “handling noise” is simultaneously encoded by the chosen form of $p(\cdot\mid \hat x)$, which determines _how_ deviations are penalized (e.g., Gaussian noise $p(x\mid\hat x)=\mathcal N(\hat x,\sigma^2I)$ yields $\mathcal L\propto \|x-\hat x\|^2$; categorical $p(x_t\mid x_{<t})$ yields cross-entropy).

---




  
### Notations on $p_\theta(x)$ and $p(x\mid \theta)$
Throughout the report, the following equivalence is used:

$$p(x \mid \theta) \;\equiv\; p_\theta(x) \;\equiv\; p(x \mid \hat x_\theta)$$

  

These notations emphasize different viewpoints:

- $p(x \mid \theta)$: likelihood / statistical conditioning:
	- The probability (density) of observing $x$, **given** model parameters $\theta$.
	- This notation emphasizes:
		- The **data-generation story**
		- Classical statistics / MLE / MAP derivations
	- Typical context:
		- Likelihood functions
		- Bayesian inference
		- Parameter estimation
- $p_\theta(x)$: family of distributions indexed by $\theta$
	- A **family of distributions indexed by** $\theta$, evaluated at $x$
	- This notation emphasizes
		- The **model as a function**
		- The **space of distributions**
		- Optimization and geometry
	- Context
		- Information geometry
		- KL divergence: $\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$
		- Generative modeling
- $p(x \mid \hat x_\theta)$: deterministic prediction + noise

    
When we are in an MLE (frequentist) setting and $\theta$ is not treated as a random variable, $p(x\mid \theta), p_\theta(x)$ are mathematically equivalent.

In **Bayesian settings**, $\theta$ is now a random variable with a prior $p(\theta)$, then we mostly use $p(x\mid\theta)$ as the likelihood, avoiding possibly ambiguous $p_\theta(x)$.


---



### Maximizing Likelihood: Parameters that make Observed Noise most Plausible

We assume a **conditional distribution** / **likelihood** / **observation model** / **noise model** :

$$p(x \mid \theta) \quad\text{or equivalently}\quad p(x \mid \hat x_\theta)$$

which answers exactly one question: Given my model’s prediction, how should deviations from it be penalized?


MLE optimizes:

$$
\hat\theta \arg\max_\theta \mathbb E_{x\sim p_{\text{data}}} [\log p(x\mid\theta)]
$$

This can be rewritten as:

$$\arg\min_\theta \mathrm{KL}\big( p_{\text{data}}(x) \;\|\; p_\theta(x) \big)
$$
But crucially:
- This is a **projection** of the unknown truth onto our model family
- Not an assumption that the truth lies inside the family


Let's define the variables formally:
- **$x$ (Observation):** The real data (noisy, stochastic).
- **$\theta$ (Parameters):** The output of our neural network $f(\cdot)$. This is **deterministic**.
- **$\varepsilon$ (Noise):** The random variable bridging the two.


We model the observation $x$ as a deterministic core plus some probabilistic noise:

$$x = \text{Model}(\theta) \oplus \text{Noise}(\varepsilon)$$

We cannot control the noise, but we can control $\theta$. Therefore, we ask:

> _"Given that I observed $x$, what must $\theta$ have been to make this observation probable?"_

This is the **Likelihood** $p(x|\theta)$.



## **6️⃣ Special case: latent-variable models (preview)**

  

In models like VAEs:

- We introduce a **prior on latent variables** p(z)
- This is **not** a prior on data
- It is an _inductive bias_ to make learning tractable

  



---



Noise answers:

- How do we penalize errors?
    
- What deviations are acceptable?
    
- How costly are outliers?
    

### MAP: distribution over parameters
  

In MAP, the prior is:
$p(\theta)$


That is:

- A distribution over **parameters**
- A belief _before seeing data_
- A regularization device


$$p(\theta) = \mathcal N(0, \lambda^{-1} I)$$

Then:

$$-\log p(\theta) = \lambda \|\theta\|^2$$

And MAP becomes:

$$\arg\min_\theta \left[ \sum \|x - \hat x\|^2 • \lambda \|\theta\|^2 \right]$$

  

This is **weight decay / L2 regularization**.

## Gaussian Noise

### Meaning
  

When we write:

$$p(x \mid \theta) = \mathcal N(x \mid \hat x_\theta, \sigma^2 I)$$

we are saying:
$$x = \hat x_\theta + \varepsilon \quad\text{where}\quad \varepsilon \sim \mathcal N(0, \sigma^2 I)$$

  

Interpretation:

- The model predicts a _mean_
- Everything the model fails to explain is lumped into $\varepsilon$
- We **pretend** $\varepsilon$ is Gaussian


  
### Leading to MSE

Gaussian likelihood:

$$-\log p(x \mid \hat x) = \frac{1}{2\sigma^2}\|x - \hat x\|^2 + \text{const}$$

Then:

$$-\log p(x \mid \theta) = \frac{1}{2\sigma^2}\|x - f_\theta(\cdot)\|^2 + \text{const}$$

  

Thus:

$$\arg\max_\theta \log p(D \mid \theta) \;\;\Longleftrightarrow\;\; \arg\min_\theta \sum \|x - \hat x\|^2$$


This is **pure MLE**.

Meaning:

- Small errors → small penalty
- Large errors → **quadratically** larger penalty
- Outliers dominate the loss

  

This defines a **preference**:  
> “I would rather make many small errors than one large one.”

---



### Common losses and their MLE interpretations

|**Loss**|**Likelihood assumption**|**Task**|
|---|---|---|
|MSE|Gaussian \mathcal N(\hat x, \sigma^2 I)|Regression|
|MAE|Laplace \text{Laplace}(\hat x, b)|Robust regression|
|Cross-entropy|Categorical / Bernoulli|Classification|
|Poisson loss|Poisson|Count data|
|Huber|Gaussian + Laplace mixture|Robust regression|

All of these are **MLE** objectives.




## **2️⃣ Loss = negative log-likelihood**

  

Formally:

$$\mathcal L(x, \hat x) \;\equiv\; -\log p(x \mid \hat x)$$


Then minimizing the loss over data is exactly:

$$\arg\max_\theta \sum \log p(x \mid \hat x_\theta)$$

which is **MLE**.

---

## **3️⃣ Canonical examples (noise → loss)**

|**Noise / likelihood assumption**|**Negative log-likelihood**|**Loss**|
|---|---|---|
|Gaussian \mathcal N(0,\sigma^2)|\propto \|x-\hat x\|^2|MSE|
|Laplace|\propto \|x-\hat x\|_1|MAE|
|Bernoulli|-x\log \hat x - (1-x)\log(1-\hat x)|Binary cross-entropy|
|Categorical|-\sum y_i \log \hat y_i|Softmax cross-entropy|
|Poisson|\hat x - x\log \hat x|Poisson loss|



  

Not **all** losses correspond to MLE.

  
Examples that **do not** directly arise from likelihoods:

- Hinge loss (SVMs)
    
- Margin losses
    
- Contrastive losses
    
- Triplet losses
    
- Many self-supervised objectives
    
- GAN discriminator loss (no explicit likelihood)
    
- Score matching (diffusion)
    

  

These optimize **other divergences or geometric criteria**, not MLE.

  

So the precise statement is:


> **All likelihood-based losses correspond to MLE under some noise model, but not all losses are likelihood-based.**

---



### **Noise model = penalty geometry**

|**Noise model**|**Loss shape**|**Behavior**|
|---|---|---|
|Gaussian|Quadratic (MSE)|Penalizes large errors heavily|
|Laplace|Linear (L1)|Robust to outliers|
|Student-t|Heavy-tailed|Very robust|
|Mixture|Multi-modal|Captures structure|

