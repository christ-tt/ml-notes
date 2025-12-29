---
tags:
  - ML
  - Theory
  - Work
---


# Data

- The **Real** Distribution: We consider supervised or self-supervised learning settings where we observe samples $x \in \mathcal{X}$ drawn i.i.d. from an **unknown data-generating distribution** $p_{\text{data}}(x)$, where we **never** assume a *parametric* form for $p_{\text{data}}$ itself.
    - In Supervised/Conditional settings, each sample $x$ is a tuple: $x = (u, y)$.
    - $u$ (Context/Input): The information we condition on (e.g., Image, Text Prompt, Features).
    - $y$ (Target/Observable): The variable we want to predict or model (e.g., Label, Next Token, Denoised Pixel).
- The **Dataset** $(\mathcal{D})$: A finite set of samples observed from the real distribution.
$$
D = \{x_1, \dots, x_N\} = \{(u_1, y_1), \dots, (u_N, y_N)\}
\sim p_{\text{data}} 
$$


# Model

We define a **Model** as a **Parametric Family of Probability Distributions**. Let 

$$
\mathcal{P} = \{p_\theta(y \mid u): \theta \in \Theta\}
$$

be a family of probability distributions indexed by parameters $\theta$ in a parameter space $\Theta$.

Our model consists of two coupled components:
- The Parameter Mapping (**Deterministic** network): A function $f_\theta$ that maps input context $u$ to distribution **parameters** $\psi$.

$$
\psi = f_\theta(u)
$$

- The Sampling Distribution (**Stochastic**): A specific probability density form $p(\cdot \mid \psi)$ that defines how the target $y$ is distributed given those parameters ($f_\theta$).

## Workflow

We never assume the parametric form of the *true* data distribution $p_{\text{data}}(y \mid u)$. Instead, we assume a *conditional* distribution $p(y \mid \psi)$, the output $y$ conditioned on our fixed parameter $\theta$ and input conditioned information (context) $u$.

This is equivalent to drawing from the distribution that we (as the modeler) assume, using *local* distribution parameters $\psi$ output by our deterministic base model.

1. **Input**($u$): The context (e.g. *"The cat sits on"*). 
2. **Network**($f_\theta$): The deterministic calculation.
3. **Sufficient Statistics**($\psi$): The result of the calculation, output parameters.
    e.g. logits of the next token over vocabularty, or probability of dog or cat [0.1, 0.9]
4. **Assumed Distribution**($p(y \mid \psi)$): the template.
    e.g. We assume: "The next word follows a Categorical distribution defined by $\psi$."
5. **Sampling**: the stochastic step; drawing from that distribution.
6. **Output**($y$): the *realization*.
    e.g. Result: next token *'mat'*, class *'dog'*...

Note that, essentially we have two types of parameters.
- $\psi$ (Local Parameters): outputs for a single data point (e.g., the mean prediction for Image #1).
- $\theta$ (Global Parameters): the weights of the network that produce $\psi$ for every data point.
Later when we introduce how we actually perform *learning*, we'll optimize on $\theta$ instead of $\psi$:
- $\psi$ is different for every single input. If we just optimized $\psi$, we would just be memorizing the dataset (setting $\mu = u$ for every point).
- $\theta$ defines the function. We want to learn $f_\theta$ that generates the correct $\psi$ for any input (including new ones). 
 
## Likelihood, Sampling Distribution, Noise, Observation Model

### Motivation: Why We need Stochastic

The necessity of introducing a stochastic part (the likelihood/noise model) essentially boils down to one fact: **The world is not a mathematical function.**
A mathematical function  is a **Many-to-One** mapping. It takes an input and produces exactly one fixed output.
However, most real-world problems are **One-to-Many**.
- Ambiguity: "The capital of..." is deterministic. "Once upon a time..." implies thousands of valid continuations. A deterministic model would collapse these into a single (likely incorrect) average.
- Unobserved Variables: We rarely observe the full state of the universe. The "Stochastic Part" captures the variations caused by hidden factors we cannot measure.
- Generative Diversity: To generate new samples (e.g., different images of dogs), we need a source of randomness to sample from.

### Terminology Equivalence

While distinct terms are used across fields, they refer to the exact same mathematcial object: the conditional probability distribution $p(y \mid \psi)$.

| Term                      | Context      | Math Form                                            | Definition & Intuition                                                                                                                    |
| ------------------------- | ------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Observation Model**     | Architecture | $y \sim \mathcal{P}(\psi)$                           | The *Blueprint*. The design choice defining the family of distribution (e.g. *"We assume the head outputs Gaussian paramaeters"*).        |
| **Likelihood**            | Training     | $\mathcal{L}(\theta) = p(y_{\text{real}} \mid \psi)$ | The *Scoreboard*. The probability density assigned to observed real data given the model's current parameters; used to compute gradients. |
| **Sampling Distribution** | Inference    | $y_\text{next} \sim p(\cdot \mid \psi)$              | The *Generator*. The probability cloud from which we draw samples to generate new data (hallucinating plausible outputs).                 |
| **Noise Model**           | Physics      | $\epsilon \sim p_\epsilon(\cdot)$                        | The *Fuzz*. The source of stochasticity, explaining why $y \neq f_\theta(x)$, defining how deviations are penalized.                      |

### Clarification: Additive Noise vs. Stochastic Sampling

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

### Conditional Independence Theorem
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

# Learning
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




# Loss
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



The goal of learning is to construct a **model** $f_\theta$ that **approximates** the data, by 
- Maximizing Likelihood: minimizing KL divergence;
- Minimizing Loss under the true data distribution.
$$
\theta^\star \approx \arg \min_\theta \mathbb E_{x \sim p_{\text{data}(x)}}\big [\mathcal L(x; \theta)\big]$$


Given a noise model, the learning objective is derived via the **negative log-likelihood**:
$$
\mathcal L(x, \hat x_\theta) \;\stackrel{\text{def}}{=}\; -\log p(x \mid \hat x_\theta).
$$
  
Thus:
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












# No Use
Equivalently, we assume:
$$x = \hat x_\theta + \varepsilon, \quad \varepsilon \sim p_\varepsilon(\cdot)$$,

where:
- $\varepsilon$ is an abstract noise variable,
- $p_\varepsilon$ is chosen by the modeler.

Noise means how we choose to interpret discrepancies between model prediction and data; different assumptions of the distribution of the noise are how we *think* we should interpret the errors. 
* For example, normal means small errors are common, large errors are increasingly unlikely; errors are symmetric, and errors should be penalized quadratically, *L2*
* Laplace noise assume mostly clean data, but occasionally huge outliers *L1* 

By assuming the conditional distribution, we are imposing our belief about the nature of the error.

If we assume Gaussian: We are saying "The model's prediction is the average, and errors are symmetric noise."

If we assume Categorical: We are saying "The world is a set of discrete choices, and the model predicts preference scores."


## Reparametrization
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


