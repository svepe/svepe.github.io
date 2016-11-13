---
layout: post
title: Variational Inference
description: introduction and derivation of ELBO maximisation
topic: inference
img: /img/vi-intro.png
---

This post is meant to be an introduction to variational inference (VI) for
people who have machine learning background, but have not studied VI in detail
before. So if you want to see what the 'evidence lower bound' is, how it is
derived and why and how to maximise it, just carry on reading.

### Introduction
Probabilisitc machine learning recasts the problem of learning into an inference
task where the observed data $\mathbb{x}$ is used to learn the distribution of
latent variables $\mathbb{z}$ that can represent either unknown parameters or
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

is often an intractable high-dimensional integral which does not have a closed
form solution. Inference methods attempt to resolve this problem by relying on
techniques such as sampling or approximations. In variational Bayesian inference
we approximate the true underlying posterior distribution
$p(\mathbb{z} | \mathbb{x})$ with another simpler distribution
$q _ \phi(\mathbb{z})$ and optimise $\phi$ such that
$q _ \phi(\mathbb{z}) = p(\mathbb{z} | \mathbb{x})$.

In the plot below, $\color{#d49f55}{q_{\phi}(z)}$ is a Gaussian distribution
which is fitted to approximate the unknown posterior $\color{#8a8a8a}
{p(\mathbb{z} | \mathbb{x})}$.

<div class="img_row">
    <img class="col three" src="/img/vi-intro.png"/>
</div>

<div class="col three caption">
  <font color="#d49f55">$q_{\phi}(z)$</font> is a Gaussian distribution which is
  fitted to approximate the unknown posterior
  <font color="#8a8a8a">$p(\mathbb{x} | \mathbb{z})$</font>
</div>

In order to fit $q_{\phi}(\mathbb{z})$ to $p(\mathbb{z} | \mathbb{x})$ we need
a distance measure $\mathcal{D}$ between the two, which we should minimise.
Thus, $\mathcal{D}$ should take 2 distributions as input arguments and provide
a single positive number which resembles the distance between the two
distributions. The branch of mathematics which deals with optimising
higher-order functions (i.e. functions which take functions as their input) is
called calculus of variations, hence the name *variational inference*. We could
use a variety of distance operators [<sup>1</sup>](#ref1) and KL-divergence is
the most widely used, defined as:

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
          p (\mathbb{z} | \mathbb{x})\right) \label{eq:dkl-argmin}
\end{align}
$$

This problem does not have a closed form solution so we need to apply some
tricks in order to come up with a stochastic optimisation procedure.

### Evidence Lower Bound (ELBO) Derivation

Let's start from equation (\ref{eq:kl-exp}):

$$
\begin{align}
\,\mathcal{D} _ {KL}
    \left(q _ {\phi}(\mathbb{z}) ||
    p (\mathbb{z} | \mathbb{x})\right)
&= \mathbb{E} _ {z \sim q _ {\phi}(\mathbb{z})}
    [\log q _ {\phi}(\mathbb{z}) -
    \underbrace{\log p (\mathbb{z} | \mathbb{x})}_
      {\log p (\mathbb{z}, \mathbb{x}) -
       \log p (\mathbb{x})}] = \nonumber \\
&= \mathbb{E} _ {z \sim q _ {\phi}(\mathbb{z})}
    [\log q _ {\phi}(\mathbb{z}) -
     \log p (\mathbb{z}, \mathbb{x})] +
     \log p (\mathbb{x}) = \nonumber \\
\nonumber \\
&= - \mathbb{E} _ {z \sim q _ {\phi}(\mathbb{z})}
    [\log p (\mathbb{z}, \mathbb{x}) -
     \log q _ {\phi}(\mathbb{z})] +
     \log p (\mathbb{x}) \nonumber \\
\nonumber \\
&= - L(\phi) + E \label{eq:elbo-simple} \geq 0
\end{align}
$$

Therefore, equation (\ref{eq:dkl-argmin}) becomes

$$
\begin{align}
\phi^*&= \underset{\phi}{\arg\min}
        \,\mathcal{D} _ {KL}
          \left(q_{\phi}(\mathbb{z}) ||
          p (\mathbb{z} | \mathbb{x})\right) \nonumber \\
&= \underset{\phi}{\arg\min}\{-L(\phi) + E\} \nonumber \\
&= \underset{\phi}{\arg\max}\{L(\phi)\} \nonumber
\end{align}
$$

Thus minimising the KL-divergence between the posterior
$p(\mathbb{z} | \mathbb{x})$ and its approximation $q_{\phi}(\mathbb{z})$ is
equivalent to maximising the quantity
$L(\phi) =\mathbb{E} _ {z \sim q _ {\phi}(\mathbb{z})}
          [\log p (\mathbb{z}, \mathbb{x}) -
           \log q _ {\phi}(\mathbb{z})]$ which is the lower bound of
the evidence $E = \log p (\mathbb{x})$ (ELBO). This can be
seen from (\ref{eq:elbo-simple}) as $E \geq L(\phi)$, or from
[<sup>2</sup>](#ref2)

$$
\begin{align}
\log p (\mathbb{x})
&= \log \int _ {Z} p(\mathbb{z}, \mathbb{x}) d\mathbb{z} \nonumber \\
&= \log \int _ {Z} p(\mathbb{z}, \mathbb{x})
                   \frac{q _ {\phi}(\mathbb{z})}
                        {q _ {\phi}(\mathbb{z})} d\mathbb{z} \nonumber \\
&= \log \mathbb{E} _ {z \sim q _ {\phi}(\mathbb{z})}
   \left[\frac{p(\mathbb{z}, \mathbb{x})}
          {q _ {\phi}(\mathbb{z})}\right] \nonumber \\
&\geq \mathbb{E} _ {z \sim q _ {\phi}(\mathbb{z})}
   \left[\log p(\mathbb{z}, \mathbb{x}) -
         q _ {\phi}(\mathbb{z})\right] \nonumber \\
\end{align}
$$

where the last step follows from Jensen's inequality [<sup>3</sup>](#ref3)
and the fact that log is a concave function. The same results can be derived
from importance sampling as well [<sup>4</sup>](#ref4).

### ELBO Maximisation


#### References
1. <a href="https://arxiv.org/abs/1610.09033" target="_blank" name="ref1">
     R. Ranganath, J. Altosaar, D. Tran, D. Blei, <i>Operator Variational Inference</i>
   </a>
2. <a href="https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf"
      target="_blank" name="ref2">
      D. Blei, <i>Notes on Variational Inference</i>
   </a>
3. <a href="https://en.wikipedia.org/wiki/Jensen%27s_inequality"
      target="_blank"
      name="ref3">
      Wikipedia, <i>Jensen's inequality</i>
   </a>
4. <a href="http://shakirm.com/papers/VITutorial.pdf"
      target="_blank"
      name="ref4">
      S. Mohamed, <i>Variational Inference for Machine Learning Tutorial</i>
   </a>


{% comment %}
<br>
{% endcomment %}
