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

## Value Function Approximation

#### Incremental Prediction Algorithms
#### MC
* the target return is $G_{t}$
* Can therefore apply supervised learning to "training data":
$$
<S_1,G_1>, <S_2,G_2>,\cdots, <S_T,G_{T}>
$$

$$
\Delta \mathbf{w} =\alpha\left(G_{t}-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)
$$


#### TD(0)
* the target is the TD target $R_{t+1} + \gamma \hat{v}(S_{t+1},\mathbf{w})$
* Can therefore apply supervised learning to "training data":
$$
<S_1,R_2 + \gamma \hat{v}(S_2),\mathbf{w})>, <S_2,R_3 + \gamma \hat{v}(S_3),\mathbf{w})>,\cdots, <S_{T-1},R_{T} + \gamma \hat{v}(S_{T},\mathbf{w})>
$$

$$
\Delta \mathbf{w} =\alpha\left(R_{t+1} + \gamma \hat{v}(S_{t+1},\mathbf{w})-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)
$$
#### For TD($\lambda$)
* the target is the $\lambda$-return $G^{\lambda}_{t}$
* Can therefore apply supervised learning to "training data":
$$
<S_1,G^{\lambda}_{1}>, <S_2,G^{\lambda}_{2}>,\cdots, <S_{T-1},G^{\lambda}_{T-1}>
$$
* Forward view 
$$
\Delta \mathbf{w} =\alpha\left(G^{\lambda}_{t}-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)
$$
* Backward view 
$$
\begin{aligned}
\delta_{t} &=R_{t+1}+\gamma \hat{v}\left(S_{t+1}, \mathbf{w}\right)-\hat{v}\left(S_{t}, \mathbf{w}\right) \\
E_{t} &=\gamma \lambda E_{t-1}+\nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)\\
\Delta \mathbf{w} &=\alpha \delta_{t} E_{t} 
\end{aligned}
$$

#### Incremental Control Algorithms
* MC, the target return is $G_{t}$
$$
\Delta \mathbf{w}=\alpha\left(G_{t}-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)
$$
* For TD(0), the target is the TD target $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$
$$
\Delta \mathbf{w}=\alpha\left(R_{t+1}+\gamma \hat{q}\left(S_{t+1}, A_{t+1}, \mathbf{w}\right)-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)
$$
* For forward-view TD($\lambda$)
$$
\Delta \mathbf{w} =\alpha\left(q^{\lambda}_{t}-\hat{q}\left(S_{t}, A_t, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_t, \mathbf{w}\right)
$$
* For backward-view TD($\lambda$)
$$
\begin{aligned}
\delta_{t} &=R_{t+1}+\gamma \hat{q}\left(S_{t+1}, A_{t+1}, \mathbf{w}\right)-\hat{q}\left(S_{t}, A_t, \mathbf{w}\right) \\
E_{t} &=\gamma \lambda E_{t-1}+\nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_t, \mathbf{w}\right)\\
\Delta \mathbf{w} &=\alpha \delta_{t} E_{t} 
\end{aligned}
$$

[Test Case: Mountain Car](../demos/reinforcement/montain_car.ipynb)

```python
  # learn with given state, action and target
  def learn(self, position, velocity, action, target):
      active_tiles = self.get_active_tiles(position, velocity, action)
      estimation = np.sum(self.weights[active_tiles])
      delta = self.step_size * (target - estimation)
      for active_tile in active_tiles:
          self.weights[active_tile] += delta

# semi_gradient_n_step_sarsa
while True:
    # go to next time step
    time += 1

    if time < T:
        # take current action and go to the new state
        new_position, new_velocity, reward = step(current_position, current_velocity, current_action)
        # choose new action
        new_action = get_action(new_position, new_velocity, value_function)

        # track new state and action
        positions.append(new_position)
        velocities.append(new_velocity)
        actions.append(new_action)
        rewards.append(reward)

        if new_position == POSITION_MAX:
            T = time

    # get the time of the state to update
    update_time = time - n
    if update_time >= 0:
        returns = 0.0
        # calculate corresponding rewards
        for t in range(update_time + 1, min(T, update_time + n) + 1):
            returns += rewards[t]
        # add estimated state action value to the return
        if update_time + n <= T:
            returns += value_function.value(positions[update_time + n],
                                            velocities[update_time + n],
                                            actions[update_time + n])
        # update the state value function
        if positions[update_time] != POSITION_MAX:
            value_function.learn(positions[update_time], velocities[update_time], actions[update_time], returns)
    if update_time == T - 1:
        break
    current_position = new_position
    current_velocity = new_velocity
    current_action = new_action
```

### Batch Reinforcement Learning
* Gradient descent is simple and appealing
* But it is not sample efficient
* Batch methods seek to find the best ftting value function
* Given the agent's experience ("training data")


#### DQN
* Take action $a_t$ at according to $\epsilon$-greedy policy
* Store transition ($s_t,a_t,r_{t+1},s_{t+1}$) in replay memory D
* Sample random mini-batch of transitions ($s_t,a_t,r,s'$)
* Compute Q-learning targets w.r.t. old, fixed parameters $w^{-}$
* Optimise MSE between Q-network and Q-learning targets
$$
\mathcal{L}_{i}\left(w_{i}\right)=\mathbb{E}_{s, a, r, s^{\prime} \sim \mathcal{D}_{i}}\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; w_{i}^{-}\right)-Q\left(s, a ; w_{i}\right)\right)^{2}\right]
$$
* Using variant of stochastic gradient descent

#### Linear Least Squares Prediction
* Using linear value function approximation $\hat{v}(s, \mathbf{w})=\mathbf{x}(s)^{\top} \mathbf{w}$
* We can solve the least squares solution directly
* At minimum of LS(w), the expected update must be zero
