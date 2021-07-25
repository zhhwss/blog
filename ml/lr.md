<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

## 逻辑回归

### 定义
逻辑回归假设数据服从伯努利分布,通过极大化似然函数的方法，运用梯度下降来求解参数，来达到将数据二分类的目的。

### 逻辑回归的基本假设

- 逻辑回归假设数据服从[伯努利分布](../statics/bernoulli_distribution.md)，伯努利分布有一个简单的例子是抛硬币，抛中为正面的概率是𝑝,抛中为负面的概率是1−𝑝.在逻辑回归这个模型里面是假设$h_\theta(x)$ 为样本为正的概率，1−$h_\theta(x)$为样本为负的概率。那么整个模型可以描述为:
$$
    \begin{split}
    P(Y=1|x)&=p=h_\theta(x) \\
    P(Y=0|x)&=1-p=1-h_\theta(x) \\
    \end{split}
$$
即
$$
    P(Y|x) =p=h_\theta(x)^Y(1-h_\theta(x))^{1-Y} 
$$



- 考虑二分类问题，给定数据集
$$
    D=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\},x_i\in R^n,y\in \{0,1\}
$$
在给定参数$\theta$下的概率为:
$$
    P(D|\theta)=\prod_{1}^N h_\theta(x_i)^{y_i}(1-h_\theta(x_i))^{1-y_i}
$$

-  逻辑回归中采用sigmoid的函数表征$x\rightarrow p$的映射,即
$$
    p=h_\theta(x)=\frac{1}{1+e^{-\theta^T x}}
$$