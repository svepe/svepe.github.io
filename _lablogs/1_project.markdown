---
layout: post
title: Variational Inference
description: introduction and derivation of ELBO maximisation
topic: inference
img: /img/vi-intro.png
---
<p align="justify">
Probabilisitc machine learning casts the problem of learning in an inference task where the observed data $\mathbb{x}$ is used to learn the distribution of latent $\mathbb{z}$ variables which can represent unknown parameters or unobservable quantities from the problem domain. Thus we are interested in finding the posterior over the latent variables

$$ p(\mathbb{z} | \mathbb{x}) = \frac{p(\mathbb{x} | \mathbb{z})p(\mathbb{z})}{p(\mathbb{x})}$$

The normalisation quantity, often referred to as marginal likelihood or evidence,

$$p(\mathbb{x}) = \int_{\boldsymbol{Z}}{p(\mathbb{x}, \mathbb{z})}\mathrm{d}\mathbb{z}$$

is often an intractible high-dimensional integral which does not have closed form solution. There are several methods for Bayesian inference and variational inference is one of them. The main idea behind variational inference is to approximate the distribution of interest, in our case $p(\mathbb{z} | \mathbb{x})$, with another simpler distribution $q(\mathbb{z})$ and make sure it is similar enough to $p(\mathbb{z} | \mathbb{x})$. In general, we consider an entire family of approximate distributions $q_{\phi}(\mathbb{z})$ parameterised by $\phi$.
</p>

<div class="img_row">
    <img class="col three" src="/img/vi-intro.png"/>
</div>
<div class="col three caption">
   Caption example!
</div>

{% comment %}
<div class="mathmode" style="color:#9a9a9a"> $$ p_{\theta}(\mathbb{z} | \mathbb{x}) $$ </div>
<div style="color:#e4af65"> $$ q_{\phi}(z) $$ </div>
This is an equation: $$ \Gamma(z) $$
 $$ \definecolor{ohra}{RGB}{228, 175, 101} $$
 $$ \color{ohra}1 + 5 = 6 $$
<div class="img_row">
    <img class="col three" src="/img/vi-intro.png"/>
</div>
<div class="col three caption">
   Caption example!
</div>
<hr>
Some text with a reference <a href="#ref1"> [1] </a>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<h3> References </h3>
<br>
1.<a href="https://nbviewer.jupyter.org/gist/AustinRochford/91cabfd2e1eecf9049774ce529ba4c16" target="+blank" id="ref1"> References </a>
{% endcomment %}
