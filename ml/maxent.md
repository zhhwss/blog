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

## 最大熵模型

首先明确最大熵模型学习的是一个概率模型，即$P(Y|X)$，最大熵模型是满足数据集约束前提下熵最大的概率模型。下面我们将分别求由**数据集导出的约束和最大熵对应的优化问题**。

### 定义

给定训练集，通过分别学习联合概率分布 $P(X,Y)$和边缘分布$P(X)$的经验分布，分别以 $\tilde{P}(X,Y))$、$\tilde{P}(X)$表示，其中：
$$
\begin{split}
    \tilde{P}(X=x,Y=y)) &= \frac{v(X=x,Y=y)}{N} \\
    \tilde{P}(X) &= \frac{v(X=x)}{N}
\end{split}
$$
其中$v(X=x,Y=y)$表示训练集中(x,y)出现的频次，N为样本个数。

在最大熵模型中，**约束条件**用特征函数$f(x,y)$度量,其描述输入$x$和输出$y$之间的某种事实，其定义如下：
$$
f(x,y)=\left\{\begin{matrix}
1 & (x,y)满足某一事实
\\ 
0 & 反之 
\end{matrix}\right.
$$
上述特征函数为一个二值函数(理论上可以取任何值)，当$x$和$y$满足该事实时取值为1，否则取值为0。这个期望其实就是约束$f$在训练集上的统计结果的均值（也就是约束$f$出现的期望的估计量）。

- 如果$f$取值为二值0,1，则表示约束$f$在训练集上出现的次数的均值;
- 如果$f$取值为任意值，则表示约束$f$在训练集上累计的结果的均值。

至此，获取到约束条件的表征，如何将特征函数（约束条件，后面不再强调）和所求模型结合在一起呢？这就需要想到如何将我们所求的经验分布和目标概率分布一一对应起来，因而有了如下操作：
* 特征函数在经验分布$\tilde{P}(X,Y))$的期望值$E_{\tilde{P}}{(f)}$:
$$
  E_{\tilde{P}}{(f)} = \sum_{x,y}\tilde{P}(x,y)f(x,y)
$$
* 特征函数关于所求模型$P(Y|X)$在经验分布$\tilde{P}(X)$的期望值$E_{P}{(f)}$
$$
  E_{P}{(f)} = \sum_{x,y}\tilde{P}(x)P(y|x)f(x,y)
$$

根据大数定理，当数据量足够并且模型优秀到获取训练集中的大部分信息时，定义这两个期望值相等：
$$
  \sum_{x,y}\tilde{P}(x)P(y|x)f(x,y) = \sum_{x,y}\tilde{P}(x)P(y|x)f(x,y)
$$
至此，我们解决了如何将所求模型与已知的经验分布相结合的问题，推导出数据的约束条件。