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

## Monte Carlo Tree Search

### Simulation
* select Tree strategy
$$
\pi_{\text {tree }}(\boldsymbol{a} \mid \boldsymbol{s})=\underset{\boldsymbol{a}}{\arg \max }\left[Q(\boldsymbol{s}, \boldsymbol{a})+c_{\text {puct }} \cdot \pi_{\phi}(\boldsymbol{a} \mid \boldsymbol{s}) \cdot \frac{\sqrt{n(\boldsymbol{s})}}{n(\boldsymbol{s}, \boldsymbol{a})+1}\right]
$$
where $s(s)=\sum_a n(s,a)$
* Expand
* Roll-out We then require an estimate of the value by the prediction of a value network $R(s_L)=V_\phi(s_L)$
* Back-up Finally, we recursively back-up the results in the tree nodes
$$
\mathrm{R}\left(\boldsymbol{s}_{i}, \boldsymbol{a}_{i}\right)=r\left(\boldsymbol{s}_{i}, \boldsymbol{a}_{i}\right)+\gamma \mathrm{R}\left(\boldsymbol{s}_{i+1}, \boldsymbol{a}_{i+1}\right)
$$
We then increment $W(s_i, a_i)$ with the new estimate $R(s_i, a_i)$, increment the visitation count $n(s_i, a_i)$ with 1, and set the mean estimate to $Q(s_i, a_i) = W(s_i, a_i)/n(s_i, a_i)$.

### Training
* Policy Network
$$
\hat{\pi}\left(\boldsymbol{a} \mid \boldsymbol{s}_{0}\right)=\frac{n\left(\boldsymbol{s}_{0}, \boldsymbol{a}\right)}{n\left(\boldsymbol{s}_{0}\right)}
$$
$$
\mathcal{L}^{\text {policy }}(\phi)=\mathrm{D}_{K L}\left(\pi_{\phi}(\boldsymbol{a} \mid s) \| \hat{\pi}(\boldsymbol{a} \mid s) \mid\right)=\mathrm{E}_{a \sim \pi_{\phi}(\boldsymbol{a} \mid \boldsymbol{s})}\left[\log \pi_{\phi}(\boldsymbol{a} \mid s)-\log \hat{\pi}(\boldsymbol{a} \mid s)\right]
$$
$$
\mathcal{L}^{H}(\phi)=H\left(\pi_{\phi}(\boldsymbol{a} \mid \boldsymbol{s})\right)=-\int \pi_{\phi}(\boldsymbol{a} \mid \boldsymbol{s}) \log \pi_{\phi}(\boldsymbol{a} \mid \boldsymbol{s}) \mathrm{d} a
$$
$$
\mathcal{L}^{\pi}(\phi)=\mathcal{L}^{\text {policy }}(\phi)-\lambda \mathcal{L}^{H}(\phi)
$$
* Value Network
$$
\hat{V}\left(\boldsymbol{s}_{0}\right)=\max _{\boldsymbol{a}} Q\left(\boldsymbol{s}_{0}, \boldsymbol{a}\right)
$$
$$
\mathcal{L}^{V}(\phi)=\mathrm{E}_{\boldsymbol{s} \sim \mathcal{D}}\left[\left(V_{\phi}(\boldsymbol{s})-\hat{V}(\boldsymbol{s})\right)^{2}\right]
$$