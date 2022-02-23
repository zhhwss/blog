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

## SAC
* add maxenxt to policy
$$
\pi^{*}=\arg \max _{\pi} \underset{\tau \sim \pi}{\mathrm{E}}\left[\sum_{t=0}^{\infty} \gamma^{t}\left(R\left(s_{t}, a_{t}, s_{t+1}\right)+\alpha H\left(\pi\left(\cdot \mid s_{t}\right)\right)\right)\right]
$$
![](images/2022-02-23-15-40-12.png)