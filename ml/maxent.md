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
首先明确最大熵模型学习的是一个概率模型$P(y|x)$，最大熵模型是满足数据集约束前提下熵最大的概率模型。下面我们将分别求由**数据集导出的约束和最大熵对应的优化问题**。

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

这里需要说明一下，在朴素贝叶斯方法中,
$$
\tilde{P}(y|x) = \frac{\tilde{P}(x,y))}{\tilde{P}(x)}
$$
是只独立估计了点$(x,y)$上的估计，并未考虑任何约束。换句话说，该方法没有考虑数据集的整体性。
综上所述，最大熵模型需要满足约束条件是：
$$
   E_{P}{(f_i)} = E_{\tilde{P}}{(f_i)},i=1,2\dots,n
$$
$n$为特征函数个数。其次$P(y|x)$还需要概率和等1的条件，如下：
$$
  \sum_y P(y|x)= 1
$$
**有了上述约束，我们可以得到最大熵模型的优化问题为**：
$$
  \begin{split}
  \max\limits_{P\in C}H(P) &= -\sum_{x,y}\tilde{P}(x)P(y|x) log P(y|x) \\
  s.t. E_{P}{(f_i)} & = E_{\tilde{P}}{(f_i)},i=1,2\dots,n \\
    \sum_y P(y|x) &= 1
  \end{split}
$$
其中$H(P) = -\sum_{x,y}\tilde{P}(x)P(y|x) log P(y|x)$是$P(y|x)$的条件熵。

### 求解
最大熵模型优化问题等价于：
$$
  \begin{split}
  \min\limits_{P\in C}H(P) &= \sum_{x,y}\tilde{P}(x)P(y|x) log P(y|x) \\
  s.t. & E_{P}{(f_i)} - E_{\tilde{P}}{(f_i)}=0,i=1,2\dots,n \\
     &  \sum_y P(y|x) = 1
  \end{split}
$$
求解上述约束最优化问题，所得解，就是最大熵模型学习的解，将约束最优化的原始问题转换为无约束最优化的对偶问题，通过求解对偶问题求解原始问题。针对上述约束最小值问题，引入拉格朗日乘子$w_0,w_1,\dots,w_n$,可得到拉格朗日函数$L(P,w)$
$$
  L(P,w)=\sum_{x,y}\tilde{P}(x)P(y|x) log P(y|x)+w_0(1-\sum_{y} P(y|x))+\sum_i^n w_i(E_{\tilde{P}}{(f_i)} - E_{P}{(f_i)}) 
$$
最优化的原始问题是：
$$
  \min\limits_{P\in C}\max\limits_{w} L(P,w)
$$
对偶问题是:
$$
  \max\limits_{w} \min\limits_{P\in C} L(P,w)
$$
由于拉格朗日函数$L(P,w)$为凸函数，原始问题的解与对偶问题的解释等价的，因此，可通过求解对偶问题求解原始问题，首先，需要求解对偶问题内部的极小化问题.首先，需要求解对偶问题内部的极小化问题:
$$
  \min\limits_{P\in C} L(P,w)
$$
则其求解是找到使函数值最小的$P_w(y|x)$ ，将其对$P(y|x)$求偏导:
$$
  \begin{split}
  \frac{\partial L(P,w)}{\partial P(y|x)}&=\sum_{x,y}\tilde{P}(x)(log P(y|x)+1)-\sum_yw_0-\sum_{x,y}(\tilde{P}(x)\sum_i^n w_i f_i(x,y)) \\
  &=\sum_{x,y}\tilde{P}(x)(log P(y|x)+1-w_0-\sum_i^n w_i f_i(x,y)))
  \end{split}
$$
其中,$\sum_{x,y}\tilde{P}(x)=\sum_{y}\sum_x\tilde{P}(x)=\sum_{y} 1$
令偏导等0，解得：
$$
  log P(y|x)+1-w_0-\sum_i^n w_i f_i(x,y)) = 0
$$
**即可得$P(y|x)$对应的最大熵形式**：
$$
  P(y|x) = exp(\sum_i^n w_i f_i(x,y) + w_0 - 1) = \frac{exp(\sum_i^n w_i f_i(x,y))}{exp(1-w_0)}
$$
此外，$P(y|x)$还需要满足约束条件：
$$
\sum_y P(y|x) = 1
$$
可得：
$$
\begin{split}
&\sum_y\frac{exp(\sum_i^n w_i f_i(x,y))}{exp(1-w_0)} = 1 \\
\Rightarrow &Z_w(x)=exp(1-w_0)=\sum_y exp(\sum_i^n w_i f_i(x,y))
\end{split}
$$
$Z_w(x)$被成为归一化因子，$P(y|x)$可以简化为：
$$
  P(y|x) = exp(\sum_i^n w_i f_i(x,y) + w_0 - 1) = \frac{exp(\sum_i^n w_i f_i(x,y))}{Z_w(x)}
$$
将$P(y|x)$带入到$L(P,w)$中可以得到关于$w$的函数$\psi(w)$，下一步求解对偶函数，即为：
$$
  \max\limits_{w} \psi(w)
$$
将其解记为$w^*$，则完成了最大熵模型的求解，总体而言，最大熵模型的学习归纳为对偶函数$\psi(w)$的求解。对于其参数的求解，利用极大似然函数，通过改进的迭代尺度法、拟牛顿法进行。


### 与逻辑回归的关系

LR是二分类的一个特例，它与最大熵模型是等价的。下面我们将证明该结论。
给定数据集，包含n个特征，样本$\vec{x}=(x_1,x_2\dots x_n)^T$，在该数据集上可定义n个特征函数如下：
$$
f_i(x,y)=\left\{\begin{matrix}
x_i & y=1
\\ 
0 & 反之 
\end{matrix}\right.
$$
将其带入到上述求解的最大熵概率模型中，有：
$$
  \begin{split}
  Z_w(x)&=\sum_y exp(\sum_i^n w_i f_i(x,y)) = exp(\sum_i^n w_i f_i(x,y=0)) + exp(\sum_i^n w_i f_i(x,y=1)) \\
  &=1+exp(\vec{w}^T\vec{x})
  \end{split}
$$
当$y=1$时:
$$
  P(y=1|x) = \frac{exp(\vec{w}^T\vec{x})}{1+exp(\vec{w}^T\vec{x})}
$$
当$y=0$时：
$$
  P(y=-|x) = \frac{1}{1+exp(\vec{w}^T\vec{x})}
$$
综上所述，当选择合适的特征函数时，LR模型和最大熵模型等价。
