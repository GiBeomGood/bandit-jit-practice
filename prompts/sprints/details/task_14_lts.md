# Linear Thompson Sampling (LTS) Conceptual Overview

Linear Thompson Sampling is a contextual bandit algorithm that balances exploration and exploitation using Bayesian inference. It assumes the expected reward of an action (arm) is a linear function of its context features.

## Mathematical Formulation
- Let $d$ be the dimension of the context features.
- At each timestep $t$, the algorithm maintains a Gaussian posterior distribution over the unknown parameter vector $\mu \in \mathbb{R}^d$.
- $B \in \mathbb{R}^{d \times d}$: Regularized design matrix (covariance inverse), initialized to $I_d$.
- $f \in \mathbb{R}^d$: Cumulative contextual reward vector, initialized to $0_d$.
- $\hat{\mu} \in \mathbb{R}^d$: Ridge regression estimate of the true parameter, given by $\hat{\mu} = B^{-1}f$.

## Algorithm Steps
For each timestep $t = 1, 2, \dots$:
1. **Sample**: Draw a parameter sample $\tilde{\mu}(t)$ from the multivariate Gaussian distribution $\mathcal{N}(\hat{\mu}, v^2 B^{-1})$. 
   - *Note on $v$*: The exploration parameter is defined as $v = R \sqrt{\frac{24}{\epsilon} d \ln(\frac{1}{\delta})}$, where $R$ is the variance proxy of sub-Gaussian error of reward, $\delta$ is the failure probability, and $\epsilon \in (0, 1)$ is an explicit hyperparameter that parametrizes the algorithm's exploration.
2. **Select Action**: For each available arm $i$ with context feature vector $b_i(t)$, compute the predicted reward $b_i(t)^T \tilde{\mu}(t)$. Play the arm $a(t)$ that maximizes this value.
3. **Observe**: Receive the actual reward $r_t$ from the environment.
4. **Update**: Update the sufficient statistics:
   - $B = B + b_{a(t)}(t)b_{a(t)}(t)^T$
   - $f = f + b_{a(t)}(t)r_t$
   - $\hat{\mu} = B^{-1}f$

This sampling mechanism naturally balances exploration and exploitation by maintaining uncertainty over the parameter $\mu$.
