---
layout: post
title: Variational Inference
description: introduction and derivation of ELBO maximisation
topic: inference
img: /img/vi-intro.png
---

Probabilisitc machine learning casts the problem of learning in an inference
task where the observed data $\mathbb{x}$ is used to learn the distribution of
latent variables $\mathbb{z}$ which can represent unknown parameters or
unobservable quantities in the problem domain. We are interested in
finding the posterior over the latent variables

$$
p(\mathbb{z} | \mathbb{x}) =
  \frac{p(\mathbb{x} | \mathbb{z})p(\mathbb{z})}
        {p(\mathbb{x})}
$$

The evidence $p(\mathbb{x})$, referred to as marginal likelihood as well,

$$
p(\mathbb{x}) =
  \int_{\boldsymbol{Z}}
        {p(\mathbb{x}, \mathbb{z})}
  \mathrm{d}\mathbb{z}
$$

is often an intractible high-dimensional integral which does not have a closed
form solution. Bayesian inference methods attempt to resolve this problem
by relying on techniques such as sampling or approximations. Variational
inference is a method for approximate Bayesian inference where the main idea
is to approximate the distribution of interest, in our case
$p(\mathbb{z} | \mathbb{x})$, with another simpler distribution
$q(\mathbb{z})$ and make sure it is similar enough to
$p(\mathbb{z} | \mathbb{x})$. In fact, we consider an entire family of
approximate distributions $q_{\phi}(\mathbb{z})$ parameterised by $\phi$. In
order to make our derivation more general we will also assume that the posterior
is parameterised as well by $\theta$, $p _ \theta (\mathbb{z} | \mathbb{x})$.

In the plot below, $\color{#d49f55}{q_{\phi}(z)}$ is a Gaussian distribution
which is fitted to approximate the unknown posterior $\color{#8a8a8a}
{p(\mathbb{x} | \mathbb{z})}$.

<div class="img_row">
    <img class="col three" src="/img/vi-intro.png"/>
</div>

<div class="col three caption">
  <font color="#d49f55">$q_{\phi}(z)$</font> is a Gaussian distribution which is
  fitted to approximate the unknown posterior
  <font color="#8a8a8a">$p(\mathbb{x} | \mathbb{z})$</font>
</div>

In order to fit $q_{\phi}(\mathbb{z})$ we need a distance measure
$\mathcal{D}$, which we should minimise, between the approximate distribution
and the true underlying one. Thus, $\mathcal{D}$ should take 2 distributions
as input arguments and provide a single positive number which resembles the
distance between the two. The branch of mathematics which deals with
optimising higher-order functions (i.e. functions which take functions as
their input) is called calculus of variations, hence the name *variational
inference*. We could use a variety of distance operators [<sup>1</sup>](#ref1)
and KL-divergence is the most widely used, defined as:

$$
\begin{align}
\mathcal{D} _ {KL} (q || p)
&= \int_{-\infty}^{\infty}q(x)log\frac{q(x)}{p(x)}dx = \nonumber \\
\nonumber \\
&= \mathbb{E} _ {x \sim q(x)}[\log q(x) - \log p(x)] = \label{eq:kl-exp}\\
\nonumber \\
&= H(q, p) - H(q) \label{eq:kl-ent}
\end{align}
$$

According to equation $(\ref{eq:kl-ent})$, $\mathcal{D} _ {KL}$ compares how
much information $q$ and $p$ convey together, $H(p,q)$, with the
information conveyed only by $q$, $H(q)$. Thus a key property is that
$\mathcal{D} _ {KL} \geq 0$ with $\mathcal{D} _ {KL} = 0 \iff H(q, p) = H(q)$.
The KL-divergence of two distributions is 0 iff $q$ and $p$ reveal no
more information than just $q$ ; in other words $q$ is "equal" to $p$, so
to speak. Therefore, in order to solve our inference problem we need to optimise

$$
\begin{align}
\phi^*= \underset{\phi}{\arg\min}
        \,\mathcal{D} _ {KL}
          \left(q_{\phi}(\mathbb{z}) ||
          p _ {\theta}(\mathbb{z} | \mathbb{x})\right)
\end{align}
$$

This problem does not have a closed form solution so we need to apply some
tricks in order to come up with a stochastic optimisation procedure.

### Evidence Lower Bound (ELBO) Derivation

Let's start from equation $(\ref{eq:kl-exp})$:

$$
\begin{align}
\,\mathcal{D} _ {KL}
    \left(q _ {\phi}(\mathbb{z}) ||
    p _ {\theta}(\mathbb{z} | \mathbb{x})\right)
&= \mathbb{E} _ {z \sim q _ {\phi}(\mathbb{z})}
    [\log q _ {\phi}(\mathbb{z}) -
    \underbrace{\log p _ {\theta}(\mathbb{z} | \mathbb{x})}_
      {\propto \, {p _ {\theta}(\mathbb{x}|\mathbb{z}) p _ {\theta}(\mathbb{z})}}] = \nonumber \\
\end{align}
$$


#### References
1. <a href="https://arxiv.org/abs/1610.09033" target="+blank" name="ref1">
     Operator Variational Inference
   </a>
2. test




{% comment %}
<br>
{% endcomment %}
