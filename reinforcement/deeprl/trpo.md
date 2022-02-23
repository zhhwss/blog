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

## TRPO
优化问题
$$
\begin{aligned}
&\operatorname{maximize}_{\theta} E_{s} \pi_{\theta_{\text {old }}}, a \pi_{\theta_{\text {old }}}\left[\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)} A_{\theta_{\text {old }}}(s, a)\right] \\
&\text { subject to } E_{s} \pi_{\theta_{\text {old }}}\left[D_{K L}\left(\pi_{\theta_{\text {old }}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s)\right)\right] \leq \delta
\end{aligned}
$$
$$
\theta_{k+1}=\theta_{k}+\alpha^{j} \sqrt{\frac{2 \delta}{g^{T} H^{-1} g}} H^{-1} g
$$
* 改进点对步长进行有效的控制
![](images/2022-02-23-15-21-51.png)