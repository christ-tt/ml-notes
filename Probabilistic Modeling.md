---
tags:
  - ML
  - Theory
  - Work
---


# **Data**
We consider supervised or self-supervised learning settings where we observe samples $x$ drawn i.i.d. from an **unknown data-generating distribution** $p_{\text{data}}(x)$. 
We **never** assume a *parametric* form for $p_{\text{data}}$.
- **The sample** $x$: In conditional settings, $x = (u, y)$.
    - $u$ (Context/Input): The information we condition on (e.g., Image, Text Prompt, Features).
    - $y$ (Target/Observable): The variable we want to predict or model (e.g., Label, Next Token, Denoised Pixel).
- The **Dataset** $D$: A finite set of samples observed from the real distribution.
$$
D = \{x_1, \dots, x_N\} = \{(u_1, y_1), \dots, (u_N, y_N)\}
\sim p_{\text{data}} 
$$

---

# **Model: The Determinisitc-Stochastic Couple**

We define a **Model** as a **Parametric Family of Probability Distributions**, indexed by parameters $\theta$ in a parameter space $\Theta$. 

$$
\mathcal{P} = \{p_\theta(y \mid u): \theta \in \Theta\}
$$

Our model architecture explicitly separtes **structure** (deterministic) from **uncertainty** (stochastic). It consists of two coupled componenets

We never assume the parametric form of the *true* data distribution $p_{\text{data}}(y \mid u)$. Instead, we assume a *conditional* distribution $p(y \mid \psi)$, the output $y$ conditioned on our fixed parameter $\theta$ and input conditioned information (context) $u$.

This is equivalent to drawing from the distribution that we (as the modeler) assume, using *local* distribution parameters $\psi$ output by our deterministic base model.

## **Deterministic Body $f_\theta$**

A function that maps input context $u$ to **Local Distribution Parameters** $\psi$.

$$
\psi = f_\theta(u)
$$

- What is $\theta$? $\theta$ represents the *learnable* parameters of the function $f$.
    - Linear Regression: $\theta = \{w, b\}$, the slope and intercept ($mx + b$)
    - Poplynomial Regression: $\theta = \{a, b, c\}$, the coefiicients ($ax^2 + bx + c$)
    - Decision Trees, $\theta$ is the split threasholds.
    - Deep Learning: $\theta = \{W_1, b_1, \dots, W_L\} billions of neural network weights.
- What is $\psi$? $\psi$ represents the *dynamic* statistics for a *specific* data point.
    - Examples: Gaussian mean/variance $[\mu, \sigma]$, or Logits over vocabulary.

Later when we introduce how we actually perform *learning*, we'll optimize on $\theta$ instead of $\psi$:
- $\psi$ is different for every single input. If we just optimized $\psi$, we would just be memorizing the dataset (setting $\mu = u$ for every point).
- $\theta$ defines the function. We want to learn $f_\theta$ that generates the correct $\psi$ for any input (including new ones). 

Just a side note, recall that in intro class, for our linear regression we introduce $b$ to handle 'noise'. Unfortunately, this was an oversimplification.
Bias $b$ handles the **Position** of the distribution, so it is part of $\mu$, while noise $\epsilon$, controls the **Spread** of the distribution, the $\sigma$.
Bias is requried to handle the case where $x = 0$ and $y \neq 0$. It also makes the function *Affine*, allowing 'cursor' to lift off the origin and shift activation functions (ReLU) left or right, which is essential for 'wiggling' the curve to fit complex shapes.
In modern deep learning, however, with the use of 
- **LayerNorm**: explicitly re-centering the data to zero and cancels out the shift provided by bias,
- **RMSNorm**: normalizing scale (variance), but empirically keeping the latent states centered around zero improves stability during training,
we usually do not use bias in MLP layers. However, in final output layer (*lm_head*), we still usually retain the bias part, learning a shift of logits.

## **The Stocastic Head $(p(\cdot \mid \psi))$**

A specific probability density form $p(\cdot \mid \psi)$ that defines how the target $y$ is distributed given those parameters ($f_\theta$).

## **Workflow**

1. **Input**($u$): The context (e.g. *"The cat sits on"*). 
2. **Network**($f_\theta$): The deterministic calculation.
3. **Sufficient Statistics**($\psi$): The result of the calculation, output parameters.
    e.g. logits of the next token over vocabularty, or probability of dog or cat [0.1, 0.9]
4. **Assumed Distribution**($p(y \mid \psi)$): the template.
    e.g. We assume: "The next word follows a Categorical distribution defined by $\psi$."
5. **Sampling**: the stochastic step; drawing from that distribution.
6. **Output**($y$): the *realization*.
    - *Training*: We **score** the real $y$ against this distribution (Likelihood).
    - *Inference*: We **sample** a new $\hat y$ from this distribution (Generation).
    e.g. Result: next token *'mat'*, class *'dog'*...

---

## **Motivation 1: Why We Need $\theta$ - The Role of Complexity**

Why we need $\theta$ at all, if the "Stochastic" part (the probability distribution) is what actually generates the data? Why not just sample directly from a distribution?
The answer lies in the distinction between **Stationary** and **Dynamic** uncertainty.

We can view our model as a simple head on top of a complex body.
- **The Head (Observation Model):** The chosen distribution form $p(\cdot \mid \psi)$ (e.g., Gaussian). By itself, this is rigid and stationary.
- **The Body (The Network ):** The deterministic function approximator $f_\theta$. This is flexible and context-aware.

Without $\theta$, our parameters $\psi$ would be constant. We would effectively be predicting the "average" of the entire universe for *every input*. The neural network solves this by making the distribution **dynamic**. Instead of a single static cloud, the distribution becomes a **cursor** that the network moves continuously across the output manifold.
By continously shifting the local distribution parameters across the high-dimensional input manifold, the model traces out a complex, non-linear shape.

**Case Study: The "Moving" Parameters**

To visualize this, suppose we want to predict **Temperature ($y$)** given **Time of Day ($x$)**.

**Case A: No $\theta$  (The Static Baseline)**
We assume a global Gaussian with fixed parameters.
$$
p(y) = \mathcal{N}(y; \mu, \sigma)
$$
- **Result:**  is a constant (e.g., 15°C). The model predicts the yearly average for every hour. It captures no structure and has huge variance.

**Case B: Linear $\theta$  (Linear Regression)**
We assume the mean moves linearly with input.
$$
p(y \mid x) = \mathcal{N}(y; \theta_1 x + \theta_0, \sigma)
$$
- **Result:** The center of the Gaussian slides along a straight line. This is better, but fails to model the actual day/night cycle (a sine wave).

**Case C: Deep Neural Network $\theta$ (The Dynamic Mover)**
We let a deep network determine the parameters.
$$
p(y \mid x) = \mathcal{N}(y; f_\theta(x), \sigma(x))
$$
* **Result:** The mean  is now the output of a multi-layer perceptron. The center of the Gaussian can trace **any arbitrary curve**, wiggling and looping to follow the true data manifold perfectly.

### **Universal Approximation Theorem (UAT)**

We rely on the Universal Approximation Theorem to guarantee that our "moving cursor" strategy is mathematically possible. UAT states that a feedforward neural network with a single hidden layer (of sufficient width) and non-linear activation functions (like ReLU) can approximate **any** continuous function $f: \mathbb{R}^n \to \mathbb{R}^m$ to arbitrary accuracy $\epsilon$.

**Implication for Modeling:**
This confirms that even if the true relationship between the input  and the distribution parameters $\psi$ is wildly complex (e.g., $\psi_{\text{true}} = g(u)$ is unknown and non-linear), there exists a configuration of weights $\theta$ such that our model $f_\theta(u)$ is virtually identical to $g(u)$.
$$
| f_\theta(u) - \psi_{\text{true}}(u) | < \epsilon
$$

**Intuition and Efficiency:**
The proof intuition for **Lipschitz continuous** functions (functions with bounded rates of change) relies on the network's ability to approximate **indicator functions of local cells**. Essentially, the network acts as a tiling mechanism: it 'queries' the function values on a fine grid and sums these local indicators (constructed via ReLUs) to reconstruct the target surface.

While generic approximation theoretically faces the **Curse of Dimensionality**—requiring an exponential number of neurons ($\approx (1/\epsilon)^d$)—neural networks circumvent this in practice. For functions with **decaying Fourier coefficients** (common in structured real-world data like images or language), the required network size scales independently of the input dimension (), explaining why deep learning remains efficient even in high-dimensional spaces.

---

## **Likelihood, Sampling Distribution, Noise, Observation Model**

### **Motivation 2: Why We need Stochastic**

The necessity of introducing a stochastic part (the likelihood/noise model) essentially boils down to one fact: 
> The world is not a **Many-to-One** mathematical function.

A **Many-to-One** mapping takes an input and produces exactly one fixed output.
However, most real-world problems are **One-to-Many**.
- Ambiguity: "The capital of..." is deterministic. "Once upon a time..." implies thousands of valid continuations. A deterministic model would collapse these into a single (likely incorrect) average.
- Unobserved Variables: We rarely observe the full state of the universe. The "Stochastic Part" captures the variations caused by hidden factors we cannot measure.
- Generative Diversity: To generate new samples (e.g., different images of dogs), we need a source of randomness to sample from.

### **Terminology Equivalence**

While distinct terms are used across fields, they refer to the exact same mathematcial object: the conditional probability distribution $p(y \mid \psi)$.

| Term                      | Context      | Math Form                                            | Definition & Intuition                                                                                                                    |
| ------------------------- | ------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Observation Model**     | Architecture | $y \sim \mathcal{P}(\psi)$                           | The *Blueprint*. The design choice defining the family of distribution (e.g. *"We assume the head outputs Gaussian paramaeters"*).        |
| **Likelihood**            | Training     | $\mathcal{L}(\theta) = p(y_{\text{real}} \mid \psi)$ | The *Scoreboard*. The probability density assigned to observed real data given the model's current parameters; used to compute gradients. |
| **Sampling Distribution** | Inference    | $y_\text{next} \sim p(\cdot \mid \psi)$              | The *Generator*. The probability cloud from which we draw samples to generate new data (hallucinating plausible outputs).                 |
| **Noise Model**           | Physics      | $\epsilon \sim p_\epsilon(\cdot)$                        | The *Fuzz*. The source of stochasticity, explaining why $y \neq f_\theta(x)$, defining how deviations are penalized.                      |

Note that we use **Likelihood** for training, and **Sampling** for inference.

In standard Supervised Learning (MLE), we do not sample during training, but in advanced Generative settings (RLHF, VAEs), we do sample during training, and we have special tricks to "learn" through that stochasticity.
- Inference (**Generation**):$\theta \to \psi \xrightarrow{\text{Sample}} \hat{y}$; We create a new, fake reality.
- Training (**Scoring**):$\theta \to \psi \xrightarrow{\text{Score}} y_{\text{real}}$; We measure the probability of the existing reality.

e.g. Top-k / Top-p / Temperature: 
- Inference-Time Heuristics for LLM decoding, not reflected / updated in training.
    - non-differentiable *"hacks"* used to chop off the tail of the distribution because the model is **imperfect**.
    - We use Top-p only because the model sometimes assigns small probability to garbage, and we want to manually force it to stay "safe."
- In RLHF (Reinforcement Learning): sampling **is** reflected in training.
    - PPO (Proximal Policy Optimization), we sample using top-p, temperature from the model during training.
    - We verify if the sampled text is "good" or "bad" using a Reward Model.
    - We then calculate a gradient to update $\theta$ so that the model becomes more likely to generate that specific "good" sample again.
    - Because sampling is discrete and breaks back-propagation, we use the **Policy Gradient Theorem (REINFORCE)** to bypass the missing gradient.

### **Clarification: Additive Noise vs. Stochastic Sampling**

It is crucial to note that "Noise" is often used as a metaphor for stochasticity, but its mathematical implementation differs by domain.
1. Continuous Models (Regression, Diffusion) Here, the noise is explicit and additive. We conceptualize the target as the deterministic prediction plus a random error term.
$$
y = f_\theta(u) + \varepsilon, \quad \text{where } \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
$$
    - The "Noise": $\varepsilon$ is a real-valued vector added to the signal.
    - Intuition: Measurement error, vibration, thermal noise.
2. Discrete Models (LLMs, Classification) Here, the target is **Categorical**. The concept of "additive noise" breaks down (e.g., you cannot calculate "Cat + $\varepsilon$"). Instead, "noise" refers to the inherent ambiguity of the sampling process.
$$
y \sim \text{Categorical}(\psi)
$$
    - The "Noise": There is no external $\varepsilon$ variable added. The stochasticity comes from the act of drawing from the distribution itself (rolling the weighted die).
    - Intuition: Aleatoric uncertainty. The input implies multiple valid outputs; the "noise" is the randomness of selection.

### **Conditional Independence Theorem**
The global model parameters $\theta$ do not influence the data $y$ directly. They only influence the data through the local distribution parameters $\psi$. We postulate the following causal chain:

$$
\theta \xrightarrow{\text{Deterministic}} \psi \xrightarrow{\text{Stochastic}} y
$$

Once the local distribution parameters $\psi$ are known, the global weights $\theta$ provide no additional information about the target $y$:

$$
y \perp \!\!\! \perp \theta \mid \psi
$$

Mathematically, this simplifies the conditional probability:

$$
p(y \mid u, \theta) \equiv p(y \mid \psi) \equiv p_\theta(y)
$$

| Viewpoint   | Name             | Meaning            | Emphasis                                                   |
| ----------- | ---------------- | ------------------ | ---------------------------------------------------------- |
| Statistical | $p(y\mid\theta)$ | Likelihood         | Probability of observing $y$ given $\theta$                |
| Geometric   | $p_\theta(y)$    | Parametric Family  | Distribution from family $\mathcal{P}$ indexed by $\theta$ |
| Mechanistic | $p(y \mid \psi)$ | Observation Model  | Probability of $y$ given statistics $\psi$                 |


---

# **MLE & MAP**
## **Frequentist viewpoint and (Log) likelihood**
From the **frequentist** perspective, probability is defined via frequencies:
- For discrete variables, probabilities are estimated by counting occurrences
- For continuous variables, probability density is inferred from samples

From our observed dataset $\mathcal{D} = \{(u_1, y_1), \dots, (u_N, y_N)\}$, where data are i.i.d. from the true distribution. 
The **Likelihood** of the dataset under our model is the joint probability:

$$
P(D \mid \theta) = \prod_{i=1}^{N} P(y_i \mid u_i, \theta)
$$

Directly optimization of this product is numerically unstable (values approaches zero). Given logarithm is monotonic, we maximize the **log-likelihood** :

$$
\log P(D \mid \theta) = \sum_{i=1}^{N} \log P(y_i \mid u_i, \theta)
$$

## Maximum Likelihood Estimation (MLE)

Given the *family of distribution* parameterized by $\theta$ 

$$
\mathcal P = \{p_\theta: \theta \in \Theta\},
$$

the MLE objective seeks the parameter $\hat{\theta}_{\text{MLE}}$ that maximizes the fit to the observed data. 

$$
\begin{align} 
\hat \theta_{\text{MLE}} &= \arg \max_\theta \log P(D\mid\theta) \\
&= \arg \max_\theta \sum_{k=1}^N \log P(x_i\mid\theta) \\
&= \arg \min_\theta \frac{1}{N}\sum_{k=1}^N -\log P(x_i\mid\theta) \\
&= \arg \min_\theta \mathbb E_{x \sim \hat p(x)} \left [- \log P(x\mid\theta) \right]
\end{align}
$$

where $\hat p(x)$ is the **Empirical Data Distribution**. Sampling from $\hat p(x)$ is equivalent to picking a data point uniformaly at random from data set $D$.
Formally, using the **Dirac** delta $\delta(\cdot)$:

$$
\hat p(x) = \frac{1}{N} \sum_{k=1}^N \delta (x - x_k)
$$ 

MLE minimizes the expected negative log-likelihood (NLL) over the observed empirical distribution.

## KL Divergence Equivalence of MLE

Minimizing Negative Log-Likelihood (NLL) is mathematically equivalent to minimizing the "distance" between the True Data Distribution and our Model.

Though we never assume a parametric form for the True Data Distribution, MLE effectively projects this complex, unknown reality onto our model family.

Let $p_{\text{data}}(x)$ be the true, unknown data distribution.
By **Law of Large Numbers**, as $N \to \infty$ , $\hat p(x) \to p_{\text{data}}(x)$. The empirical expectation converges to the true expectation: 

$$
\mathbb E_{x \sim \hat p(x)}[\cdot] \;\xrightarrow{N \to \infty}\; \mathbb E_{x \sim p_{\text{data}}(x)}[\cdot]
$$

The optimization problem becomes minimizing the **Cross-Entropy** between *Truth* and *Model*: 

$$
\hat \theta_{\text{MLE}} = \arg \min_\theta \mathbb E_{x \sim p_{\text{data}}(x)} \left [- \log p(x\mid\theta) \right] 
$$

We can expand the KL Divergence term:

$$
\begin{align}
\mathrm{KL}\big(p_{\text{data}} \,\|\, p_\theta\big) &= \int p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p(x \mid \theta)} dx\\ &= \mathbb E_{p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(x)}{p(x\mid\theta)} \right] \\
&= \underbrace{\mathbb E_{p_{\text{data}}}[\log p_{\text{data}}(x)]}_{\text{Entropy } H(p_{\text{data}})} - \underbrace{\mathbb E_{p_{\text{data}}}[\log p(x\mid\theta)]}_{\text{Likelihood}}
\end{align}
$$

Since the Entropy of the true data $H(p_\text{data})$ is a constant with respect to our model parameters $\theta$, minimizing KL Divergence is identical to maximizing Likelihood.

$$
\hat \theta_{\text{MLE}} = \arg \min_\theta \mathrm{KL}\big(p_{\text{data}}(x) \,\|\, p(x\mid \theta)\big)
$$

We can also establish this using basic logarithm and linearity of expectation 

$$
\begin{align*} \log (\frac{A}{B}) &= \log(A) - \log (B) \\ \mathbb E[-\log A] & = \mathbb E[\log B - \log A - \log B] \\ &= \mathbb E[\log\frac{B}{A}] - \mathbb E[\log B] \\ \mathbb E_{p_{\text{data}}}[-\log p(x\mid\theta)] &= \mathbb E_{p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(x)}{p(x\mid\theta)} \right] - \mathbb E_{p_{\text{data}}}[\log p_{\text{data}}(x)] \\ &= \mathrm{KL}\big(p_{\text{data}}(x) \,\|\, p(x\mid \theta)\big) + H(p_{\text{data}})\end{align*}
$$

MLE minimizes the **Forward KL Divergence** from the true data distribution to the model distribution.
- This effectively forces the model to *cover* the data.
- Heavily penalizes the model for assigning low probability to real data points (missing modes).
- Does not heavily penalize assigning probability to regions where data doesn't exist (potential for hallucinations).


### **When Data is Scarce: Maximum A Posteriori (MAP)**

When the dataset is small, MLE can overfit, memorizing noise in $\hat p(x)$ rather than learning $p_\text{data}$.
In these cases, we adopt a **Bayesian** perspective by introducing a **Prior** distribution $P(\theta)$ over the parameters themselves.

Using Bayes’ rule:

$$
P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)} \;\;\propto\;\; P(D \mid \theta) P(\theta)
$$

The MAP objective maximizes the **Posterior** probability of the parameters:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta \log P(D \mid \theta) + \log P(\theta)
$$

In practice, the choice of the prior (e.g. Gaussian, Laplace) leads directly to *regularization* terms (like Weight Decay), which we will discuss in later sections.

---

# **Loss**

In deep learning engineering, we often treat *Loss functions* as different *measuring* sticks, e.g. 
    - *Mean Squared Error* measures distance.
    - *Cross Entropy* measures surprise.
From the probabilistic perspective, however, there's only **ONE** loss function: *Negative Log Likelihood* (NLL).

$$
\mathcal{L}(\theta) = - \log p(y \mid \psi)
$$

Every specific loss (MSE, MAE, CE) is simply the NLL derived from a specific choice of **Observation Model** (Noise Distribution).

## **Cross-Entropy: Discrete, Categorical Targets**

**Model outputs** a probability vector, *logits*, via softmax, given context $u$,
$$
\psi = s_\theta(u) \in \mathbb R^K, \qquad s_\theta(u) = \left(s_{\theta,1}(u), \dots, s_{\theta,K}(u)\right),
$$
mapped to probabilities
$$
\begin{align*}
\pi_\theta(u) &= \text{softmax}(s_\theta(u)) \\
\pi_{\theta, k}(u) &= \frac{\exp(s_{\theta,k}(u)}{\sum_{j=1}^K \exp(s_{\theta, j}(u)}, k = 1, \dots, K
\end{align*}
$$

**Assuming categorical likelihood**, target $y \in \{1, \dots, K\}$ (e.g., Next Token), $y \mid u \sim\text{Categorical}(\pi_\theta(u))$
$$
\begin{align*}
p_\theta(y=k\mid u)&=\pi_{\theta,k}(u) \\
p_\theta(y \mid u) &= \prod_{k=1}^K \pi_{\theta, k}(u)^{\mathbf 1[y=k]} \\
-\log p_\theta(y\mid u) &= -\log \pi_{\theta,y}(u) \\
&= -s_{\theta,y}(u) + \log\!\left(\sum_{j=1}^K e^{s_{\theta,j}(u)}\right)
\end{align*}
$$

For a target **one-hot** distribution $q(\cdot)$ (ground truth), cross-entropy is:
$$
H(q,\pi_\theta) \triangleq -\sum_{k=1}^K q_k \log \pi_{\theta,k}(u).
$$
If $q$ is one-hot at class $y$, this reduces exactly to categorical NLL:
$$
\ell_{\text{CE}}(u,y;\theta)= -\log \pi_{\theta,y}(u) = -\log p_\theta(y\mid u).
$$
So cross-entropy is exactly NLL for a categorical likelihood.

LLM special case (sequence): for tokens $x_{1:T}$,
$$
\mathcal L_{\text{LLM}}(\theta)= -\sum_{t=1}^T \log p_\theta(x_t\mid x_{<t}),
$$
i.e., a sum of categorical NLL terms.

### **Minimizing Cross-Entropy is Equivalent to Minimizing KL**
Minimizing CE minimizies KL from the lablel's Dirac distribution to the model distribution. Recall,
$$
\mathrm{KL}(q\|p) = H(q,p) - H(q) \qquad H(q) = -\sum_k q_k \log q_k
$$
For a **Dirac / one-hot** distribution, the entropy is zero, $q^{(y)}=0$, so
$$
\mathrm{KL}\!\left(q^{(y)} \,\|\, \pi_\theta(u)\right)
= H\!\left(q^{(y)},\pi_\theta(u)\right)
= -\log \pi_{\theta,y}(u)
= -\log p_\theta(y\mid u).
$$

### **Bernoulli $\to$ Binary Cross-Entropy (BCE)**

**Assumption**: Target $y \in \{0, 1\}$; model outputs probability $\psi = p$
$$
\begin{align*}
p(y \mid \psi) &=  \psi^y(1-\psi)^{1-y} \\
- \log p(y \mid \psi) &= -\left(y\log \psi + (1 - y) \log(1 - \psi)\right)
\end{align*}
$$

- Standard **Binary Cross-Entropy** used in *logistic regression* and discriminators.
- This is equivalent to Categorical CE case with 2 classes.

## **MSE & MAE: Continuous Targets (Gaussian / Laplace)**

### **Gaussian Noise $\to$ Mean Squared Error (MSE)**

**Assumed Likelihood**: For continuous targets $y \in \mathbb{R}^d $, we use *additive* isotropic Gaussian noise:
$$
y = \mu_\theta(u) + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2I).
$$
Equivalently, 
$$
y \mid \mu \sim \mathcal{N}(\mu, \sigma^2I), \qquad p_\theta(y \mid u) = \mathcal{N}(y \mid \mu_\theta(u), \sigma^2I).
$$

**NLL**:
$$
\begin{align}
p(y \mid \mu, \sigma^2I) &= \frac{1}{(2\pi\sigma^2)^{d/2}}\exp \left(-\frac{\|y-\mu\|^2}{2\sigma^2}\right) \\
- \log p(y \mid \mu, \sigma^2) &= \frac{d}{2}\log(2\pi \sigma^2) + \frac{1}{2\sigma^2}\|y-\mu\|^2
\end{align}
$$
We usually assume $\sigma$ is fixed (not learned), so the first term becomes the constant w.r.t. $\mu$ (and hence w.r.t. $\theta$. Also, the scaling $\frac{1}{2\sigma^2}$ is also constant, so minimizing NLL is equivalent to minimizing *squred error*.
$$
\ell_{\mathrm{NLL}}(u,y;\theta) = -\log p_\theta(y\mid u) \;\propto\; \|y-\mu_\theta(u)\|^2 = \ell_{\mathrm{MSE}}(u,y;\theta).
$$

- Minimizing NLL is equivalent to minimizing **L2** Distance (MSE);
- Large errors are penalized quadratically.
- The model estimates the conditional **mean** $\mathbb{E}[y \mid \mu]$

### **Laplace Noise $\to$ Mean Absolute Error (MAE)**

**Assumption**: The target is drawn from a Laplace distribution (sharper peak, heavier tails).

$$
\begin{align}
p(y\mid \mu) &= \frac{1}{2b}\exp\left(-\frac{|y-\mu|}{b}\right) \\
-\log p(y \mid \mu) &\propto |y - \mu| + \text{const}
\end{align}
$$

- Minimizing NLL is equivalent to minimizing the **L1** Distance (MAE).
- The model estimates to the conditional **Median**.
- Robust to outliers because the penalty grows linearly, not quadratically.

---

## **Unification via Empirical Risk**

We can now unify all these seemingly different formulas under the Empirical Risk Minimization (ERM) framework derived from MLE.The Objective:
$$
\theta^* = \arg \min_\theta \frac{1}{N} \sum_{i=1}^N \underbrace{\mathcal{L}(y_i, f_\theta(u_i))}_{\text{Loss Term}} + \underbrace{\lambda \mathcal{R}(\theta)}_{\text{Regularization}}
$$
1. The Loss Term (Data Fit)
    As shown above, the "Loss Term" is exactly the negative log-likelihood of our noise assumption:
    $$
    \mathcal{L}(y, \psi) \equiv -\log p_{\text{Noise}}(y \mid \psi)
    $$
    If we assume Gaussian $\to$ we get MSE. If we assume Categorical $\to$ we get Cross-Entropy.
2. The Regularization Term (Prior)
    This connects back to MAP (Maximum A Posteriori). The regularization term corresponds to the negative log-prior of the parameters:
    $$
    \mathcal{R}(\theta) \equiv -\log p_{\text{Prior}}(\theta)
    $$
    Example: L2 Regularization (Weight Decay)
    - **Assumption**: Weights follow a Gaussian Prior $\theta \sim \mathcal{N}(0, \tau^2 I)$.
    - **Derivation**:
    $$
    -\log p(\theta) \propto \|\theta\|^2
    $$
    - **Result**: Adding weight_decay to the optimizer is mathematically equivalent to placing a Gaussian prior on your network weights.

**Summary**: Distribution $\to$ LossTask

| Task                | Target $y$               | Assumed Likelihood $p(y\mid \psi)$   | Loss                      |
| ------------------- | ------------------------ | ------------------------------------ | ------------------------- |
| **Regression**      | Real Scalar $\mathbb{R}$ | Gaussian $\mathcal{N}(\mu, I)$       | MSE (L2)                  |
| **Robust Reg.**     | Real Scalar $\mathbb{R}$ | **Laplace** $\text{Laplace}(\mu, b)$ | **MAE** (L1)              |
| **Binary Class.**   | $\{0, 1\}$               | **Bernoulli**                        | **Binary Cross-Entropy**  |
| **LLM / Class.**    | $\{1, \dots, K\}$        | **Categorical**                      | **Cross-Entropy**         |
| **Metric Learning** | Vectors                  | **Categorical** (over batch)         | **Contrastive (InfoNCE)** |

---

# **Learning Paradigms & Case Studies**

Having established that a model is simply a parametric tool to approximate a probability distribution, we can now classify distinct learning paradigms based on **what** distribution they model and **how** they access data.

## **The Taxonomy**

### **Supervised vs. Unsupervised (The Data View)**

This distinction is based on the **Dataset $\mathcal{D}$**.

1. **Supervised Learning:**
* **Data:** Pairs $x = (u, y)$ .
* **Goal:** Learn the conditional distribution $p(y \mid u)$.
* *Analogy:* "Teacher and Student." The teacher provides the question ($u$) and the correct answer ($y$).


2. **Unsupervised Learning:**
* **Data:** Only raw samples $x = (u)$. (Or effectively $y=u$).
* **Goal:** Learn the structure of the data itself, often modeled as $p(u)$.
* *Analogy:* "Explorer." The model observes the world and tries to find patterns without explicit guidance.
* *Note:* **Self-Supervised Learning** (like training LLMs) is technically Unsupervised (we only have text), but we mathematically frame it as Supervised by creating artificial $(u, y)$ pairs from the data itself (e.g., $u =$ "The", $y=$"cat").



### **Discriminative vs. Generative (The Modeling View)**

This distinction is based on the **Probability Distribution** being learned.

1. **Discriminative Models:**
* **Model:** $p(y\mid u)$ 
* **Focus:** The **Decision Boundary**. Given input $u$, which output $y$ is most likely?
* *Behavior:* Can distinguish "Dog" from "Cat," but cannot draw a dog.
* *Examples:* Logistic Regression, Image Classifiers.


2. **Generative Models:**
* **Model:**$p(u,y)$ (Joint) or $p(u)$ (Marginal).
* **Focus:** The **Data Geometry**. How is the data actually created?
* *Behavior:* Can generate new samples $u_{\text{new}} \sim p(u)$. To do this, they must understand the internal structure of the data.
* *Examples:* LLMs, Diffusion Models, VAEs.

---

## **Case Studies: The Unified Workflow**

We will now dissect distinct architectures using our unified 4-step workflow:

1. **Input ():** The context.
2. **Body ():** The deterministic universal approximator (Neural Network).
3. **Head ():** The local distribution parameters output by the body.
4. **Sampling ():** The assumed stochastic process.

---

### **Case I: Classic Regression (Supervised / Discriminative)**

* **Task:** Predict House Price () given Features ().
* **Assumption:** Errors are normally distributed.

1. **Input:** Vector of features  (footage, location...).
2. **Body:** MLP or Linear Layer.



*(Note: We often assume fixed variance , so the net only outputs the mean).*
3. **Assumed Dist:** Gaussian.


4. **Loss (Training):** Negative Log-Likelihood  **MSE**.


5. **Inference:**



---

### **Case II: Image Classification (Supervised / Discriminative)**

* **Task:** Predict Class Label () given Image ().
* **Assumption:** One mutually exclusive category.

1. **Input:** Image pixels  (e.g., ).
2. **Body:** ResNet / ViT (Vision Transformer).
* Outputs a vector of raw scores (logits) .
* .


3. **Assumed Dist:** Categorical.


4. **Loss (Training):** NLL  **Cross-Entropy**.


5. **Inference:**



---

### **Case III: Large Language Models (Self-Supervised / Generative)**

* **Task:** Predict Next Token () given Context ().
* **Assumption:** The next token is a categorical choice from a vocabulary .

1. **Input:** Sequence of token IDs .
2. **Body:** Transformer Decoder (e.g., Llama, GPT).
* Maps sequence to a hidden state .
* Projects  to vocabulary size (e.g., 32k logits).
* .


3. **Assumed Dist:** Categorical (over Vocabulary).


4. **Loss (Training):** NLL  **Cross-Entropy** (on the correct next token).
5. **Inference (Decoding):**
* Here, we **do** sample.
* **Strategies:** Greedy (), Top-k (sample from top ), Top-p (Nucleus sampling).
* *Goal:* Balance correctness (mode seeking) with diversity (tail sampling).



---

### **Case IV: Image Generation (Diffusion Models)**

* **Task:** Generate Image () given Text () and Noise.
* **Assumption:** The reverse process of adding noise is a Gaussian denoising step.

1. **Input:**
* Noisy Image .
* Time step .
* Text Prompt  (Conditioning).


2. **Body:** U-Net or DiT (Diffusion Transformer).
* The network predicts the **noise** component .
* Mathematically, predicting noise  is equivalent to predicting the mean  of the denoised image distribution.
* .


3. **Assumed Dist:** Gaussian.


4. **Loss (Training):** NLL  **MSE** (between predicted noise and real noise).


5. **Inference:**
* Start with pure noise .
* Iteratively sample  using the predicted .



---

### **Case V: Multi-Modal Understanding (CLIP / Contrastive)**

* **Task:** Align Images () and Text () in a shared space.
* **Assumption:** Correct pairs  should have high similarity; mismatched pairs  should have low similarity.

1. **Input:** Batch of  image-text pairs .
2. **Body:** Two Encoders.
* .
* .
* .


3. **Assumed Dist:** Categorical (Batch-wise).
* For the -th image, which of the  texts is the correct one?
* .


4. **Loss (Training):** **InfoNCE** (Symmetric Cross-Entropy over the batch).
5. **Inference (Zero-Shot Classification):**
* Embed input image.
* Embed "classes" as text (e.g., "A photo of a dog", "A photo of a cat").
* Pick the text embedding with highest cosine similarity.



---

## **Summary Table: The Grand Unification**

| Model | Input  | Body  | Head Parameter  | Assumed Dist. | Loss (NLL) |
| --- | --- | --- | --- | --- | --- |
| **Regression** | Features | MLP | Mean  | **Gaussian** | MSE |
| **Classification** | Image | ResNet | Logits | **Categorical** | Cross-Entropy |
| **LLM** | Tokens | Transformer | Logits | **Categorical** | Cross-Entropy |
| **Diffusion** | Noisy Img | U-Net | Mean  (via noise) | **Gaussian** | MSE |
| **CLIP** | Img/Txt | Dual Enc. | Sim. Matrix | **Categorical** | InfoNCE |


















# LATER



### Input and Output

**Input:** The "Conditioning Information."
**Output:** The "Distribution Parameters"

| Task | Input | Output |
| --- | --- | --- |
| **Supervised (Regression)** | **Features** (e.g., House Size) | **Predicted Mean Target** (e.g., Price ) |
| **Supervised (Classification)** | **Image** (e.g., Pixels) | **Logits / Probabilities** (Vector) |
| **Autoencoder** | **Image** () | **Reconstructed Image** () |
| **VAE (Decoder)** | **Latent Code** () | **Reconstructed Mean** () |

## Case Study
### 1. Large Language Models (LLM)

*Task: Text Generation (e.g., GPT-4, Llama)*

* **Input:** A sequence of tokens (Context).
* `"The capital of France is"`


* **The Model ():** A Transformer Decoder.
* **Output ():** **Logits**.
* A vector of size 50k (vocabulary size). Each number represents the "unnormalized score" for a word.
* 


* **Likelihood Assumption (Noise):** **Categorical Distribution**.
* We assume the next word is chosen probabilistically (rolling a 50,000-sided die).
* To get probabilities, we apply Softmax to .


* **Loss:** Cross-Entropy (Negative Log-Likelihood).
* We want the probability of the *actual* next word ("Paris") to be 1.0.



> **Key Insight:** The "Noise" here is the ambiguity of language. The model outputs a distribution because there is rarely only *one* valid next word.

---

### 2. Vision-Language Models (Image Understanding)

*Task: Visual Question Answering (VQA) or Captioning (e.g., Llava, GPT-4V)*

* **Input:** An Image + A Text Prompt.
* `



* "What is the animal doing?"`

* **The Model ():** A Visual Encoder (ViT) connected to an LLM.
* **Output ():** **Logits** (Same as LLM!).
* Even though the input is multi-modal, the *output* is usually just text tokens.
*  predicts the token "sleeping".


* **Likelihood Assumption:** **Categorical Distribution**.
* **Loss:** Cross-Entropy.

> **Key Insight:** To the model, an image is just "translated" into vectors that look like text embeddings. The objective remains "predict the next word," but conditioned on visual features.

---

### 3. Text-to-Image Generation (Diffusion Models)

*Task: Generating images from prompts (e.g., Stable Diffusion, Midjourney)*

This is the most complex one because the "Signal" and "Noise" are flipped.

* **Input:** A **Noisy Image** () + A **Text Prompt** + Time step ().
* We take a clean image and add 50% static to it. We give this static-filled image to the model.


* **The Model ():** A U-Net or Transformer.
* **Output ():** **The Noise ()**.
* Strangely, the model is trained to predict *what the static looks like*, so we can subtract it.


* **Likelihood Assumption:** **Gaussian**.
* We assume the pixel values of the clean image were corrupted by Adding Gaussian Noise.
* Therefore, the probability of the previous clean state depends on a Gaussian distribution.


* **Loss:** **Mean Squared Error (MSE)**.
* 
* We measure the Euclidean distance between the *actual* static we added and the *static* the model guessed.



> **Key Insight:** In Diffusion,  acts as a "Denoising Engine." By predicting the noise (assumed Gaussian), it allows us to reverse the process and uncover the image.

---

### 4. Image-Text Matching (Contrastive Learning)

*Task: Connecting Images to Concepts (e.g., CLIP)*

This is not "generating" data, but "aligning" distributions.

* **Input:** A batch of Images  and a batch of Text Captions .
* **The Model ():** Two Encoders (Image Encoder & Text Encoder).
* **Output ():** **Embedding Vectors**.
*  (Vector summary of the image)
*  (Vector summary of the text)


* **Likelihood Assumption:** **InfoNCE (Categorical over the batch)**.
* We assume that for a specific image , the "correct" caption  is the one that aligns best, and all other captions in the batch are "noise/distractors."
* It treats the batch like a multiple-choice test.


* **Loss:** Contrastive Loss.
* Maximize the dot product (similarity) of correct pairs .
* Minimize the dot product of incorrect pairs.



---

### Summary Table

| Domain | Input to  | Output  (Parameters) | Assumed Distribution (Noise) | Loss Function |
| --- | --- | --- | --- | --- |
| **LLM** | Text Context | Logits (Next Token Scores) | **Categorical** (Multinomial) | Cross-Entropy |
| **Image Understanding** | Image + Text Query | Logits (Next Token Scores) | **Categorical** (Multinomial) | Cross-Entropy |
| **Diffusion (Generation)** | Noisy Image + Prompt | Predicted Noise () | **Gaussian** (Normal) | MSE () |
| **CLIP (Matching)** | Image + Text | Embedding Vectors | **Categorical** (over batch) | Contrastive Loss |


  



---

---


## **4. Conceptual Distinctions**

### **4.1. Generative vs. Discriminative**

* **Discriminative (Classifiers/CLIP):** Approximates . It learns the **decision boundary** between classes. It ignores the density of .
* **Generative (LLMs/Diffusion/VAE):** Approximates . It learns the **manifold geometry** and density of the data. This requires "learning the noise" structure to sample valid new data.

### **4.2. "Learning the Noise"**

* In **Homoscedastic** regression (Standard MSE), we assume noise is constant and ignore it.
* In **Probabilistic** models (LLMs/VAEs),  predicts the parameters of the noise distribution (e.g., variance/spread), effectively treating uncertainty as part of the signal.
To answer your questions directly:

1. **Are they all self-supervised?** Mostly, yes (for pre-training), but Fine-Tuning often introduces human labels.
2. **Are they learning the distribution or the boundary?** Generative models (LLM, Diffusion) learn the **Distribution**. Discriminative models (CLIP, Classifiers) learn the **Boundary**.

Here is the breakdown of why this matters.

### 1. Self-Supervised vs. Supervised

The confusion usually comes from the definition of a "Label."

* **Supervised:** The label  is external information provided by a human (e.g., "This image contains a cat").
* **Self-Supervised:** The label  is a hidden part of the input data  itself. The model plays "Fill in the Blanks."

| Model | Type | Where does the "Label" come from? |
| --- | --- | --- |
| **LLM (Pre-training)** | **Self-Supervised** | The "next word" is already in the text. We just hide it and ask the model to guess. No human needed. |
| **Diffusion** | **Self-Supervised** | The "noise" is mathematically generated by us. We take a clean image, add noise, and tell the model "predict the noise we just added." |
| **CLIP (Matching)** | **Weakly Supervised** | Uses (Image, Text) pairs scraped from the web (alt-text). While not hand-labeled by annotators strictly for training, the text *is* an external label describing the image. |
| **VQA / Chatbot (Fine-Tuning)** | **Supervised** | Humans explicitly write: *Input: "Summarize this." Output: "Here is the summary..."* This is **Instruction Tuning**. |

> **Key Nuance:** Pre-training is usually self-supervised (learning the structure of the world). Fine-tuning (RLHF, SFT) is supervised (learning to follow instructions).

### 2. Learning Distribution () vs. Decision Boundary ()

This is the difference between an **Artist** and a **Critic**.

#### A. The "Artist": Learning the Distribution (Generative)

**Models:** LLMs, Diffusion, VAEs.
**Goal:** They want to know **probability density** everywhere.

* They don't just want to know "Is this a cat?"
* They need to know "What does a cat look like?" (Where is the manifold of cat images?).
* **Why?** To generate a new cat, they need to sample from high-probability regions of that distribution.
* **Mathematically:** They approximate  (or ).

#### B. The "Critic": Learning the Boundary (Discriminative)

**Models:** Classifiers, Detectors, CLIP.
**Goal:** They want to find the **line** that separates concepts.

* They don't care if the input is a realistic cat or a cartoon cat, as long as it's on the "Cat" side of the line.
* They ignore the internal structure of the data manifold and focus only on the differences between classes.
* **Mathematically:** They approximate  directly, often ignoring .

### Comparison Visualized

* **Discriminative (Boundary):** Draws a line between Red and Blue dots. It doesn't care where the dots are, just which side they are on.
* **Generative (Distribution):** Circles the region where Red dots live and the region where Blue dots live. It knows the *shape* of the data clusters.

### Summary Table

| Model Class | Examples | Supervision Source | What it Learns |
| --- | --- | --- | --- |
| **Generative** | LLM, Diffusion, VAE | **Self-Supervised** (Data predicts itself) | **The Distribution:** The shape, density, and structure of the data manifold. |
| **Discriminative** | ResNet (ImageNet), Spam Filter | **Supervised** (Human labels) | **The Boundary:** The surface separating Class A from Class B. |
| **Hybrid** | CLIP (Contrastive) | **Web-Supervised** (Alt-text) | **Alignment:** It learns a "Joint Space" where boundaries between images align with boundaries between text. |

