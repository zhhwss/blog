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

## XGBoost
XGBoost和[GBDT](./gbdt.md)类似，都是求解模型：
$$
    f_M(x)=\sum_{i=1}^M h_m(x)
$$
它与GBDT的主要不同之处在于：
- **XGBoost在求解子树的时候，对目标函数做了二阶泰勒展开**
- XGBoost目标函数显式的加入了正则项
- 在构造子树的时的目标即为最小化展开后的目标函数，而GBDT是用子树首先拟合目标函数梯度，然后求解最优的叶子节点值。

### XGBoost 的损失函数
首先，定义求解第m课子树的损失函数为：
$$
    L_m = \sum_{i=1}^n L(y_i, f_{m-1}(x) + h_m(x_i)) + \gamma J + \frac{\lambda}{2}\sum_{j=1}^J w_{jm}^2
$$
其中, $n$是样本个数, $h_m(x)$是要求解的子树, $J$是$h_m$叶子节点个数, $w_{jm}$是叶子节点的值。
其中$\gamma J + \frac{\lambda}{2}\sum_{j=1}^J w_{jm}^2$是正则项。
我们可以将损失函数的第一行进行二阶泰勒展开，即为：
$$
    L_m \approx \sum_{i=1}^{n}\left(L\left(y_{i}, f_{m-1}\left(x_{i}\right)\right)+\frac{\partial L\left(y_{i}, f_{m-1}\left(x_{i}\right)\right.}{\partial f_{m-1}\left(x_{i}\right)} h_{m}\left(x_{i}\right)+\frac{1}{2} \frac{\partial^{2} L\left(y_{i}, f_{m-1}\left(x_{i}\right)\right.}{\partial f_{m-1}^{2}\left(x_{i}\right)} h_{m}^{2}\left(x_{i}\right)\right)+\gamma J+\frac{\lambda}{2} \sum_{j=1}^{J} w_{m j}^{2}
$$
为了方便，我们把第i个样本在第m个弱学习器的一阶和二阶导数分别记为：
$$
    g_{mi} = \frac{\partial L\left(y_{i}, f_{m-1}\left(x_{i}\right)\right.}{\partial f_{m-1}\left(x_{i}\right)}, h_{mi}=\frac{\partial^{2} L\left(y_{i}, f_{m-1}\left(x_{i}\right)\right.}{\partial f_{m-1}^{2}\left(x_{i}\right)}
$$
则我们的损失函数现在可以表达为：
$$
L_m \approx \sum_{i=1}^{n}\left(L\left(y_{i}, f_{m-1}\left(x_{i}\right)\right) + g_{mi} h_m(x_i) + h_{mt}h_{m}^{2}(x_{i}) \right) +\gamma J+\frac{\lambda}{2} \sum_{j=1}^{J} w_{m j}^{2}
$$
损失函数里面$L\left(y_{i}, f_{m-1}\left(x_{i}\right)\right)$是常数，对最小化无影响，可以去掉，同时由于每个决策树的第$js$个叶子节点的取值最终会是同一个值$w_{mj}$,因此我们的损失函数可以继续化简。
$$
\begin{aligned}
L_{m} &\left.\approx \sum_{i=1}^{n} g_{m i} h_{m}\left(x_{i}\right)+\frac{1}{2} h_{m i} h_{m}^{2}\left(x_{i}\right)\right)+\gamma J+\frac{\lambda}{2} \sum_{j=1}^{J} w_{m j}^{2} \\
&=\sum_{j=1}^{J}\left(\sum_{x_{i} \in R_{m j}} g_{m i} w_{m j}+\frac{1}{2} \sum_{x_{i} \in R_{m j}} h_{m i} w_{m j}^{2}\right)+\gamma J+\frac{\lambda}{2} \sum_{j=1}^{J} w_{m j}^{2} \\
&=\sum_{j=1}^{J}\left[\left(\sum_{x_{i} \in R_{m j}} g_{m i}\right) w_{m j}+\frac{1}{2}\left(\sum_{x_{i} \in R_{m j}} h_{m i}+\lambda\right) w_{m j}^{2}\right]+\gamma J
\end{aligned}
$$
我们把每个叶子节点区域样本的一阶和二阶导数的和单独表示如下：
$$
    G_{mj}=\sum_{x_{i} \in R_{m j}} g_{m i}, H_{mj}=\sum_{x_{i} \in R_{m j}} h_{m i}
$$
最终损失函数的形式可以表示为：
$$
    L_m = \sum_{j=1}^{J} \left[G_{mj}w_{mj}+\frac{1}{2}\left(H_{mj}+\lambda\right)w_{mj}^2\right]+\gamma J \tag{1}
$$
XGBoost构建子树的目的就是最小化上述函数。

### 求解
最小化公式(1)需要：
- 构建一颗最优子树结构
- 优化该最优子树结构叶子节点的值

实际上，在构造子树的过程中，我们需要一个指标用来挑选分裂特征和分裂节点。这里，因为我们要最小化公式(1)，因此可以将公式(1)直接作为指标。然而，使用公式(1)作为指标的时候，我们需要先求解当前节点分裂后子节点对应的最优值$w_{mj}$，带入公式(1). **因此，我们首先要求解任意分裂节点的最优节点值。**

这个问题实际很简单，对于第一个问题，其实是比较简单的，我们直接基于损失函数对$w_{mj}$求导并令导数为0即可:
$$
    \frac{L_m}{w_{mj}}=G_{mj}+(H_{mj}+\lambda)w_{mj} = 0
$$
因此,对$w_{mj}$的最优解为：
$$
    w_{mj} = -\frac{G_{mj}}{H_{mj}+\lambda}
$$
**有了叶子节点的最优值，我们将进一步看如何挑选分裂特征和分裂点**
将最优$w_{mj}$带入到式(1)中，有：
$$
    L_m= -\frac{1}{2}\sum_{j=1}^{J}\frac{G_{mj}^2}{H_{mj}+\lambda}+\gamma J
$$
如果我们每次做左右子树分裂时，可以最大程度的减少损失函数的损失就最好了。也就是说，假设当前节点左右子树的一阶二阶导数和为$G_L,H_L,G_R,H_L$, 则我们期望最大化下式：
$$
-\frac{1}{2} \frac{\left(G_{L}+G_{R}\right)^{2}}{H_{L}+H_{R}+\lambda}+\gamma J-\left(-\frac{1}{2} \frac{G_{L}^{2}}{H_{L}+\lambda}-\frac{1}{2} \frac{G_{R}^{2}}{H_{R}+\lambda}+\gamma(J+1)\right)
$$
整理下上式后，我们期望最大化的是：
$$
\max \frac{1}{2} \frac{G_{L}^{2}}{H_{L}+\lambda}+\frac{1}{2} \frac{G_{R}^{2}}{H_{R}+\lambda}-\frac{1}{2} \frac{\left(G_{L}+G_{R}\right)^{2}}{H_{L}+H_{R}+\lambda}-\gamma
$$
因此XGBoost的挑选分裂特征和分裂点就是枚举待选特征和特征的分裂点以最大化上式。