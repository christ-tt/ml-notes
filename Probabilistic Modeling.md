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
From the probabilistic persepctive, however, there's only **ONE** loss function: *Negative Log Likelihood* (NLL).

$$
\mathcal{L}(\theta) = - \log p(y \mid \psi)
$$

Every specific loss (MSE, MAE, CE) is simply the NLL derived from a specific choice of **Observation Model** (Noise Distribution).

## **Continous Targets: Regression**

When the target is continuous $y\in \mathbb{R}$, we typically model the uncerntainty as the *additive* noise.

### **Gaussian Noise $\to$ Mean Squared Error (MSE)**

**Assumption**: The target is drawn from the Normal Distribution centered at the model's prediction $\mu$, with fixed variance $\sigma^2 = 1$.

$$
\begin{align}
y \mid \mu &\sim \mathcal{N}(\mu, 1)\\
p(y \mid \mu) &= \frac{1}{\sqrt{2\pi}}\exp \left(-\frac{(y-\mu)^2}{2}\right) \\
- \log p(y \mid \mu) &= -\log(\frac{1}{\sqrt{2\pi}}) - \frac{-(y - \mu)^2}{2} \\
&= \frac{1}{2}\|y - \mu\|^2 + \text{constant}
\end{align}
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

## **Discrete Targets: Classification & Generation**

When the target is discrete, "noise" is the stochasticity of selecting a *category*.

### **Bernoulli $\to$ Binary Cross-Entropy (BCE)**



















# LATER

### When MLE reduces to Mean Squared Error (MSE)
Assume the model: $$P(x \mid \theta) = \mathcal{N}(x \mid \mu_\theta, \sigma^2 I) $$
The negative log-likelihood for a single sample is, let $c$ be constant, $$- \log P(x\mid \theta) = c \frac{1}{2\sigma^2}\|x-\mu_\theta\|^2$$
Thus, $$\arg\max_\theta \log P(D \mid \theta) \;\;\Longleftrightarrow\;\; \arg\min_\theta \sum_{k=1}^N \|x_k - \mu_\theta\|^2$$
Minimizing mean squared error is equivalent to MLE under isotropic Gaussian noise assumption, which is also the reason why MSE appears ubiquitously in regression, autoencoders, and reconstruction losses.



## Mapping Function
> Mapping *'What we know'* to *The Parameters of the Distribution*.

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




> **A loss function is simply a negative log-likelihood under an assumed noise model.**

| **Noise model** p(x \mid \hat x) | **Interpretation**  | **Resulting loss**    |
| -------------------------------- | ------------------- | --------------------- |
| \mathcal N(\hat x, \sigma^2 I)   | Gaussian noise      | Mean Squared Error    |
| Laplace(\hat x, b)               | Heavy-tailed noise  | L1 loss               |
| Bernoulli(\hat x)                | Binary outcomes     | Binary cross-entropy  |
| Categorical(\hat x)              | Multiclass outcomes | Softmax cross-entropy |

- MSE Loss - Gaussian Noise, the *function* is the conditional *mean* $$\begin{align*} \mathcal L(y, \hat y) &= \|y - \hat y\|^2 \\ f^*(\mu) &= \mathbb E[y\mid \mu] \end{align*}$$
- MAE Loss - Laplace Noise, the *function* is the conditional *median* $$ \begin{align*} \mathcal L(y, \hat y) &= |y - \hat y| \\ f^*(\mu) &= \text{median}(y\mid \mu) \end{align*}$$
- Cross-Entropy Loss, the network approximates the *entire conditional distribution* $$ \begin{align*} \mathcal L(y, \hat y) &= -\log \hat p(y) \\ f^*(\mu) &= p_{\text{data}}(y \mid \mu) \end{align*}$$

i.e. our model approximates a target mapping that is implicitly defined by the data distribution and the learning objective.


By maximizing likelihood, we are essentially finding the parameters that make the observed noise most plausible.






  



---

---


#  Learning - Types (Supervised/Un-), MLE, MAP
We classify modern architectures based on their input/output structure and implied noise assumptions.

### **3.1. Traditional Supervised Learning**

* **Goal:** Learn the conditional boundary .
* **Data:** Explicit pairs  provided by humans.

| Task | Input | Output () | Assumed Noise | Loss |
| --- | --- | --- | --- | --- |
| **Regression** | Features | Mean Value  | Gaussian | MSE |
| **Classification** | Image/Text | Logits | Categorical | Cross-Entropy |

### **3.2. Self-Supervised Generative Learning (LLMs & Vision)**

* **Goal:** Learn the joint distribution  (often factorized as sequence ).
* **Data:** Unlabeled . The target is derived from the input itself (masking/shifting).

#### **Case Study A: Large Language Models (LLM)**

* **Input:** Context tokens .
* **Output ():** Logits vector (Vocabulary Size ).
* **Noise Assumption:** Categorical. The next word is sampled probabilistically.
* **Mechanism:**  approximates the conditional distribution of language.

#### **Case Study B: Diffusion Models (Image Gen)**

* **Input:** Noisy Image  + Time  + Prompt.
* **Output ():** The predicted Noise .
* **Noise Assumption:** Gaussian. We assume the corruption process adds Normal noise.
* **Mechanism:** Minimizing  (MSE) allows us to reverse the noise process.

### **3.3. Multi-Modal Learning (CLIP / VLM)**

* **Goal:** Align distributions of different modalities.

| Model | Input | Output () | Noise Assumption | Loss |
| --- | --- | --- | --- | --- |
| **VLM (e.g., GPT-4V)** | Image + Text | Next Token Logits | Categorical | Cross-Entropy |
| **CLIP (Alignment)** | Image + Text | Embeddings  | Categorical (Batch) | Contrastive (InfoNCE) |

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

