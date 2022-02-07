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

## Model Free Prediction

* Estimate the value function of an unknown MDP

### Monte-Carlo Reinforcement Learning

* 目标, 从历史回放中学习$v_\pi$, $S_1,A_1,R_2,\cdots,S_k\sim \pi$
* return $G_t=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{T-1}R_{t+T}$
* $v_\pi(s)=E[G_t|S_t=s]$
* Monte-Carlo policy evaluation uses empirical mean return instead of expected return

#### First-Visit Monte-Carlo Policy Evaluation
![](images/2022-02-07-19-14-27.png)
#### Every-Visit Monte-Carlo Policy Evaluation
![](images/2022-02-07-19-15-07.png)
#### Incremental Monte-Carlo Updates
$$
\begin{aligned}
&N\left(S_{t}\right) \leftarrow N\left(S_{t}\right)+1 \\
&V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\frac{1}{N\left(S_{t}\right)}\left(G_{t}-V\left(S_{t}\right)\right)
\end{aligned}
$$
In non-stationary problems,
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}-V\left(S_{t}\right)\right)
$$

[Test Case: Blackjack](../demos/reinforcement/blackjack.ipynb)

```python
# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    states_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_no_usable_ace_count = np.ones((10, 10))
    for i in tqdm(range(0, episodes)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count

```


### Temporal-Difference Learning
* TD learns from incomplete episodes, by bootstrapping
* TD updates a guess towards a guess

####  Comparison of MC and TD
* MC: $V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}-V\left(S_{t}\right)\right)$
* TD(0): $V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right)$
* TD target: $R_{t+1}+\gamma V\left(S_{t+1}\right)$
* TD error: $R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)$
![](images/2022-02-07-19-42-36.png)
![](images/2022-02-07-19-43-53.png)