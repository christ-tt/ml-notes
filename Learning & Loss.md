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

Here is the formalized skeleton for your technical report. It structures the transition from "Curve Fitting" (Engineering view) to "Maximum Likelihood Estimation" (Probabilistic view) and applies it across modern AI architectures.

---

# **Technical Report: The Probabilistic Foundations of Deep Learning**

## **1. The Learning Framework**

This section establishes the mathematical notation and the relationship between the deterministic neural network and the stochastic world it models.

### **1.1. The Data Universe**

* **The Real Distribution ():** We assume observations  are drawn i.i.d. from an unknown, complex distribution . We never assume a parametric form for  itself.
* **The Dataset ():** A finite set of samples  observed from the real distribution.

### **1.2. The Model Architecture ()**

We define the neural network as a deterministic function mapping inputs to distribution parameters.

* ** (Model Parameters):** The weights and biases of the neural network.
* **Input ():** The conditioning information (e.g., input features, text context, noisy image).
* **Output ():** The deterministic output of the network. Crucially,  is **not** the prediction of the data itself, but the **parameter** of the conditional distribution (Likelihood).



### **1.3. The Bridge: Noise and Likelihood**

We connect the deterministic output  to the stochastic observation  via a Noise Model.

* **Noise ():** The stochastic component we cannot predict. The form of the noise (Gaussian, Bernoulli, Categorical) is an **assumption** we make about the data generation process.
* **Likelihood ():** The probability of observing the data  given the model's output . This is effectively "The probability of the noise required to bridge the gap between  and ."



---

## **2. The Trinity: Loss, Likelihood, and Noise**

This section provides the proof that "Minimizing Engineering Loss" is identical to "Maximizing Probabilistic Likelihood."

### **2.1. The Core Objective**

We seek to find parameters  that maximize the probability of the observed data:



Equivalently, we minimize the Negative Log-Likelihood (NLL):


### **2.2. Derivation: Regression  Gaussian Noise**

* **Assumption:** The noise is additive and Gaussian: , where .
* **Likelihood:** 
* **Loss Derivation:**


* **Conclusion:** Minimizing **Mean Squared Error (MSE)** is equivalent to Maximum Likelihood Estimation (MLE) under a **Gaussian Noise** assumption.

### **2.3. Derivation: Classification  Categorical Noise**

* **Assumption:** The data follows a Categorical (Multinoulli) distribution (e.g., word selection).
* **Likelihood:** 
* **Loss Derivation:**


* **Conclusion:** Minimizing **Cross-Entropy Loss** is equivalent to MLE under a **Categorical/Multinomial** assumption.

---

## **3. Taxonomy of Learning Paradigms**

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

---

## **5. Summary Definitions Table**

| Term | Definition | Role in Learning |
| --- | --- | --- |
| **** | The Neural Network | The deterministic function approximator. |
| **** | Model Output | The **parameters** of the likelihood distribution (e.g., Mean, Logits). |
| **Noise** | Stochastic Process | The assumed bridge between ideal  and real . |
| **Likelihood** |  | The probability of the data under the assumed noise model. |
| **Loss** |  | The engineering metric. . |



You are hitting on the fundamental distinction between **Generative** and **Discriminative** modeling.

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

### Conclusion

* **LLMs & Diffusion** are primarily **Self-Supervised** and learn the **Distribution**. This allows them to create new content.
* **Classifiers & CLIP** are **Supervised** (or web-supervised) and learn the **Boundary**. This allows them to categorize or retrieve existing content.

### Next Step

This wraps up the theory of VAEs, Likelihood, and Generative vs. Discriminative models. Would you like to switch gears to the **implementation side**? We can look at a simple **PyTorch implementation of a VAE** to see how the Encoder, Decoder, and Reparameterization Trick () are actually written in code.


Here is a breakdown of how the framework  applies to modern Multi-modal and Generative AI.

In all these cases, remember:

1. : The Deep Neural Network (Transformer/UNet).
2. **Input:** What the model sees.
3. **Output ():** The parameters (logits, means) of the distribution.
4. **Likelihood/Loss:** The assumption about how the data varies (Noise).

---

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

### Final Note on "Approximation"

* In **LLMs**, we approximate the conditional distribution of **Language**: .
* In **Diffusion**, we approximate the conditional distribution of **Visual Noise**: .

Both rely on  to output the parameters of a distribution that makes the training data "most likely."


Here is the breakdown of the notation to make it concrete.

### 1. What are the Input and Output of ?

The function  is the **deterministic core** of your neural network. Its job is to map "What we know" to "The parameters of the distribution."

**The Input:** The "Conditioning Information."
**The Output ():** The "Distribution Parameters" (usually the Mean).

Here is how it changes depending on the task:

| Task | Input to  | Output of  () |
| --- | --- | --- |
| **Supervised (Regression)** | **Features** (e.g., House Size) | **Predicted Mean Target** (e.g., Price ) |
| **Supervised (Classification)** | **Image** (e.g., Pixels) | **Logits / Probabilities** (Vector) |
| **Autoencoder** | **Image** () | **Reconstructed Image** () |
| **VAE (Decoder)** | **Latent Code** () | **Reconstructed Mean** () |

### 2. Definition of 

In the context of Likelihood and Noise,  is the **Pure Signal**.

* **Mathematically:** 
* **Intuitively:**  is what the model believes the data *should* be, before the real world adds noise to it.
* **Statistically:**  serves as the **Location Parameter** (Mean) of the likelihood distribution.

### 3. Why is ?

This is a statement about **Sufficient Statistics** and conditional independence.

* **The Chain of Events:**
To determine the probability of a data point , the process flows like this:


* **The Logic:**
*  is the global "recipe" book (the neural net weights).
*  is the specific "meal" cooked for *this specific input*.
* Once the meal () is cooked, the recipe book () doesn't matter anymore for the taste test ().


* **The Math:**
If we assume a Gaussian noise model with variance :


Since , we can substitute directly:


Therefore, the likelihood of the data given the weights is exactly the likelihood of the data given the model's output.

### Summary Visualization

Think of an Archer (The Model):

1. ** (Weights):** The archer's muscle memory and skill.
2. **Input:** The wind speed and distance to target.
3. ** (Output):** Where the archer *aims*. This is a single, deterministic point.
4. **Noise:** The arrow wobbles in the air.
5. ** (Observation):** Where the arrow actually lands.

We judge the archer () by measuring the distance between where they aimed () and where the arrow landed ().

### Next Step

Would you like to formalize this into a "Cheat Sheet" for your notes that maps every component of the VAE (Encoder, Decoder, Latent, Loss) to this specific  and  framework?
