
<style type="text/css" rel="stylesheet">
code {
overflow: auto;
white-space: pre-wrap !important;
word-wrap: break-word !important;
}
</style>

## 伯努利分布

伯努利分布（Bernoulli Distribution），是一种离散分布，又称为 “0-1 分布” 或 “两点分布”。例如抛硬币的正面或反面，物品有缺陷或没缺陷，病人康复或未康复，此类满足「只有两种可能，试验结果相互独立且对立」的随机变量通常称为伯努利随机变量。

我们1概率为p(0≤p≤1)，则0概率为q=1-p，则概率密度函数为：

$$
P(x)=p^x(1-p)^{(1-x)} = \left\{\begin{matrix} p & x = 1  \\ 1 - p & x = 0\end{matrix}\right.
$$

其期望值为:

$$
\int xP(x)dx = E[xP(x)] = 1\times p + 0 \times (1-p) = p
$$

方差为:

$$
    Var(x) = E[(x-E(x))^2]=(1-p)^2 \times p + (0-p)^2 \times (1-p) = p(1-p)
$$

**成功（出现1值）1次需要实验的平均次数为**

$$
n=1\times p + 2\times (1-p)p+ 3\times (1-p)^2p+\dots=\sum_{i=1}^{\infty}i\times(1-p)^{i-1}p=\frac{1}{p}
$$

## 二项分布
二项分布（Binomial Distribution）也是一种**离散型概率分布**，又称为「n 重伯努利分布」。其定义如下

```
如果随机变量序列 X_n(n=1, 2, …) 中的随机变量均服从与参数为 p 的伯努利分布，那么随机变量序列 X_n 就形成了参数为 p 的 n 重伯努利试验。例如，假定重复抛掷一枚均匀硬币 n 次，如果在第 i 次抛掷中出现正面，令 X_i=1；如果出现反面，则令 X_i=0。那么，随机变量 X_n(n=1, 2, …) 就形成了参数为 1/2 的 n 重伯努利试验。
```

进行n次这样的试验，成功了x次，则失败次数为n-x，发生这种情况的概率可用下面公式来计算：

$$
    P(x)=C_{n}^xp^x(1-p)^{n-x}
$$

其中$C_{n}^x$是组合公式，定义如下 :

$$
    C_{n}^x = \frac{n!}{x!(n-x)!}
$$

由于n次实验之间相关独立，因此二项分布的均值和方差分别为：

$$
    \begin{split}
    E(x) &= \sum_{i=1}^n E(X_i) = np \\
    Var(x) &= \sum_{i=1}^n Var(X_i) = np(1-p)
    \end{split}
$$
