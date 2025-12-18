---
tags:
  - Research
themes: Image Generation, Multi-Modal, Machine Learning
created: 2025-12-14
---
# Generation
## Distribution of Data

**Hypothesis**: The data are distributed about a curved or otherwise non-linear **manifold** in high dimensional space
* Assuming data lives in $x \in R^D$ where $D$ very large, but the true degrees of freedom are much smaller: $\mathcal{M} \subset R^D$    where $\text{dim} (\mathcal M) \ll D$  
	* Examples: 
		* Images: lighting, pose, identity, texture
		* Speech: phonemes, pitch, speaker identity
		* Text embeddings: semantics, syntax, style
	* Random points in $R^D$ are meaningless - only points near $\mathcal M$ look like data.
* The principal components of all instances of the target class of data lie on this manifold.
- To generate data for this class, we muse select a point on this manifold.



**Problems**: 
* Characterizing the manifold.
* Having a good strategy for selecting points from it.


### Auto Encoders
* An autoencoder captures a manifold because **reconstruction pressure** forces the encoder to discover the **minimal** set of **coordinates** that locally parameterize the data, collapsing all off-manifold directions.
* ![[VAE.png]]
* Encoder **Compression**: By forcing the high-dimensional input $x$ through a low-dimensional bottleneck $z$ , the  encoder must *discard* information
* Decoder **Reconstruction**: Learning a mapping from the low-dimensional coordinates back to the high-dimensional space, reconstructing only points near the data distribution
* Formally, an autoencoder learns $$ \min_{\theta, \phi} \mathbb E(x) \sim p_{\text{data}}\left[||x - D_\phi (E_\theta (x))||^2\right]$$ where 
	* Encoder: $E: \mathbb R^D \to \mathbb R^d$ 
	* Decoder: $D: \mathbb R^d \to \mathbb R^D$ 
	* Bottleneck: $d < D$ 
* Around any data point, the manifold looks approximately linear, and the reconstruction loss enforces
	* Sensitivity along tangent directions: Let $x' = x + \epsilon v, v\in T_x\mathcal M$  
		* Then 
			* $x'$ is still a valid data point
			* Reconstruction loss must be low for both $x$ and $x'$ 
		* Therefore
			* Encoder must **distinguish** $x$ and $x'$
			* Latent representation must change meaningfully
	* Insensitivity along normal directions: Let $x' = x + \epsilon n, n\perp T_x\mathcal M$ 
		* Then
			* $x'$ is off-manifold
			* No training data exists there
		* Best strategy for minimizing expected loss:
			* Map $x'$ back toward the nearest point on $\mathcal M$ 
		* This is implicit denoising, even without explicit noise.
* Formally,
	* Encoder Jacobian spans the tangent space
		* Row View: the rows of the encoder's Jacobian represent the gradient vectors of the latent units; pointing the directions where $z$ changes the most, as $z$ only changes when we move along the manifold (tangent).
	* Decoder maps latent points back onto $M$ 
		* Column View: the columns of the decoder's Jacobian represent the partial $\frac{\partial \hat x}{\partial z_i}$, forming a basis for the tangent plane at $\hat x$.
* So,
	* Small changes in latent space $\to$ realistic variations
	* Small off-manifold perturbations $\to$ collapsed by encoder.

**Problem with AEs**
* Improper choice of input to the decoder can result in incorrect generation
* How do we know what inputs are reasonable for the decoder?
	* Only choose input ($z$'s) that are typical of the class
		* I.E. drawn from the distribution of $z$'s. (But what is this distribution?)

### Manifold (Briefly)
* Geometric Intuitions: The 'Flat Earth' Analogy
	* A manifold is a shape that looks *curved* when viewed from *far away* (globally), but looks flat when you view a tiny patch of it up *close* (locally).
	* Fundamental property of smooth manifold: zooming in effectively infinite times on a curve (1D) or a surface (2D), the curvature disappears, indistinguishable from a line or a plane.
	* Mathematically, we can approximate the manifold $M$ around a point $x$ using **Tangent Space** $(T_x\mathcal M)$ , a linear hyperplane.
* Tangent v.s. Normal Directions
	* Tangent: vectors lying on the manifold surface. Moving $x$ in this direction changes the data content meaningfully (e.g. rotating a digit slightly)
	* Normal: vectors pointing *away* from the manifold (orthogonal). Moving $x$ in this direction adds noise/artifacts that make the data invalid.