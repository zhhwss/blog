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

**成功（出现1值）的平均次数为**

$$
n=1\times p + 2\times (1-p)p+ 3\times (1-p)^2p+\dots=\sum_{i=1}^{\infty}i\times(1-p)^{i-1}p=\frac{1}{p}
$$