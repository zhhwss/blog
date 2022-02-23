<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
    	MathJax.Hub.Config({tex2jax: {
             inlineMath: [['$','$']],
             displayMath: [["\\(","\\)"],["\\[","\\]"]],
             processEscapes: true
           }
         });
    </script>
</head>

## Exploration and Exploitation
- `Exploitation` Make the best decision given current information
- `Exploration` Gather more information
- Naive Exploration
  - Add noise to greedy policy (e.g. $\epsilon$-greedy)
- Optimistic Initialisation
  - Assume the best until proven otherwise
- Optimism in the Face of Uncertainty
  - Prefer actions with uncertain values
- Probability Matching
  - Select actions according to probability they are best
- Information State Search
  - Lookahead search incorporating value of information


### Greedy and $\epsilon$-greedy algorithm
* $\epsilon$-Greedy Algorithm, With probability $\epsilon$ select a random action
  * Constant $\epsilon$ ensures minimum regret
$$
I_{t} \geq \frac{\epsilon}{\mathcal{A}} \sum_{a \in \mathcal{A}} \Delta_{a}
$$
  * $\epsilon$-greedy has linear total regret
* Optimistic Initialisation
  * Simple and practical idea: initialise Q(a) to high value
  * Update action value by incremental Monte-Carlo evaluation
  * Encourages systematic exploration early on
  * But can still lock onto suboptimal action
* Decaying $\epsilon_t$-Greedy Algorithm
  * has logarithmic asymptotic total regret
  * Unfortunately, schedule requires advance knowledge of gaps

### Upper Confidence Bound
* Estimate an upper confidence $\hat{U}_t(a)$ for each action value
* Such that $Q(a)\le \hat{Q}_t(a) + \hat{U}_t(a)$ with high probability
* Select action maximising Upper Confidence Bound (UCB)
$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \hat{Q}_{t}(a)+\hat{U}_{t}(a)
$$
* We will apply Hoeffding's Inequality to rewards of the bandit conditioned on selecting action a
$$
\mathbb{P}\left[Q(a)>\hat{Q}_{t}(a)+U_{t}(a)\right] \leq e^{-2 N_{t}(a) U_{t}(a)^{2}}
$$
* This leads to the UCB1 algorithm
$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(a)+\sqrt{\frac{2 \log t}{N_{t}(a)}}
$$

### Bayesian Bandits
* Bayesian bandits exploit prior knowledge of rewards $P[R]$
* They compute posterior distribution of rewards  $P[R|h_t]$
* Use posterior to guide exploration
  * Upper confidence bounds (Bayesian UCB)
  * Probability matching (Thompson sampling)

#### UCB
* Assume reward distribution is Gaussian, $R_a=N(r;\mu_a,\sigma_a^2)$
* Compute Gaussian posterior over $\mu_a$ and $\sigma_a^2$
$$
p\left[\mu_{a}, \sigma_{a}^{2} \mid h_{t}\right] \propto p\left[\mu_{a}, \sigma_{a}^{2}\right] \prod_{t \mid a_{t}=a} \mathcal{N}\left(\mid r_{t} ; \mu_{a}, \sigma_{a}^{2}\right)
$$
* Pick action that maximises standard deviation of Q(a)
$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(a)+ c\sigma_a/\sqrt{N_t(a)}
$$

#### Probability Matching
* Probability matching selects action a according to probability that a is the optimal action
$$
\pi\left(a \mid h_{t}\right)=\mathbb{P}\left[Q(a)>Q\left(a^{\prime}\right), \forall a^{\prime} \neq a \mid h_{t}\right]
$$
* Probability matching is optimistic in the face of uncertainty
  * Uncertain actions have higher probability of being max

#### Thompson Sampling
* Thompson sampling implements probability matching
$$
\begin{aligned}
\pi\left(a \mid h_{t}\right) &=\mathbb{P}\left[Q(a)>Q\left(a^{\prime}\right), \forall a^{\prime} \neq a \mid h_{t}\right] \\
&=\mathbb{E}_{\mathcal{R} \mid h_{t}}[\mathbf{1}(a=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(a))]
\end{aligned}
$$
* Use Bayes law to compute posterior distribution $p[R|h_t]$
* Sample a reward distribution R from posterior
* Compute action-value function $Q(a) = E[R_a]$
* Select action maximising value on sample, $a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(a)$

#### Information State Space
* At each step there is an information state $\hat{s}$
  * $\hat{s}$ is a statistic of the history, $\hat{s}_t=f(h_t)$
  * summarising all information accumulated so far
* Each action a causes a transition to a new information state $\hat{s}'$  (by adding information), with probability $\hat{P}_{\hat{s}\hat{s}'}^a$
e.g.
* The information state is $\hat{s}=<\alpha,\beta>$
  * $\alpha_a$ counts the pulls of arm a where reward was 0
  *  $\beta_a$ counts the pulls of arm a where reward was 1