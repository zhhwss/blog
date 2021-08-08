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

## GBDT
梯度提升树（Grandient Boosting）是提升树（Boosting Tree）的一种改进算法。简而言之就是用boosting的方式求解模型：
$$
    f_M(x)=\sum_{i=1}^M h_m(x)
$$
使得loss函数在数据集D上最小：
$$
    \argmin_{f_M} L(y, f_M)
$$
如果求解$f_M$即为GDBT的关键。首先我们看一下提升树。

### 提升树算法
1. 初始化$f_0(x) = 0$
2. 对于$m=1,2,3\cdots M$
   a. 计算残差项
   $$
        r_{mi} = y_i - f_{m-1}(x_i),\quad i=1,\cdots n
   $$
   其中n是训练集样本个数
   b. 拟合残差$r_{mi}$学习一个回归树，得到$h_m(x)$
   c. 更新$f_m(x) = f_{m-1}(x)+h_m(x)$
3. 得到回归树
$$
    f_M(x)=\sum_{i=1}^M h_m(x)
$$

这里的关键点就是每次构建残差项，并构建一个子回归树来拟合这个残差项。

到这里，就会引出一个问题：
- **对于分类问题，该如何构建子树**，使用残差项效率肯定非常低
<!-- - **如果单纯最小化残差项导致另一个问题，即很容易过拟合** -->

解决上述问题的办法是我在boosting的构建子树的时候，**目标不是拟合残差，而是使得loss函数最小**。

那么如果使loss函数最小呢？这就是GBDT需要解决的问题。

### GBDT 算法
如何最小化损失函数呢？
我们首先回忆一下最速下降法（梯度下降法）最小化函数的过程：
通过一阶泰特展开证明负梯度是函数下降最快的方向：
$$
    f(\theta_{k+1}) \approx f(\theta_{k})+\frac{\partial f(\theta_k)}{\partial \theta_k}(\theta_{k+1} - \theta_{k})
$$
则优化函数f时：
$$
\theta_{k+1} = \theta_{k} - \eta\frac{\partial f(\theta_k)}{\partial \theta_k}
$$
GBDT很上述过程类似。
$$
    \begin{split}
    L(y, f_{m}) &\approx  L(y, f_{m-1})+\frac{\partial L(y, f_{m-1})}{\partial  f_{m-1}}(f_{m} - f_{m-1}) \\
    f_{m}&=f_{m-1} - \eta \frac{\partial L(y, f_{m-1})}{\partial  f_{m-1}}
    \end{split}
$$
因此GBDT的整体流程如下所示：
- 首先，求解loss当前的负梯度
$$
    r_{mi} = -\frac{\partial {L(y_i, f_{m-1})}}{\partial f_{m-1}}, \quad i=1,\cdots n
$$
- 然后用一颗子树去拟合$r_{mi}$. 这里构建子树的结果是负梯度相近的样本会被分到相同的叶子节点中. 我们假设子树$h_m(x)$的叶子节点区域为$R_{jm},\quad j=1,\cdots J$, $J$是叶子节点的个数.
- 求解完梯度后, 就需要求步长$\eta$. 这里, 我们可以直接把它和梯度的乘积等价成叶子节点区域$R_{jm}$对应的值.
  - 如果我们用子树来预测$r_{mi}$, 还是需要乘上$\eta$才能得到想要的结果$f_{m}=f_{m-1} + \eta \times r_{mi}$, 其中$h_{m}(x) = \eta \times r_{mi}$为当前子树
  - 归约到相同叶子节点的样本, 在预测阶段被认为具有相同的$\eta r_{mi}$
  - 因此, 我们可以直接用$\eta r_{mi}$代替叶子节点的值
综上所述, 可以等价为优化问题
$$
\gamma_{j m}=\arg \min _{\gamma} \sum_{x_{i} \in R_{j m}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\gamma\right), \quad j \in \{1,\cdots, J\} \tag{1}
$$
- 更新学习器
$$
    f_{m}(x) = f_{m-1} = \sum_{j=1}^J \gamma_{jm}I(x\in R_{jm})
$$
- 重复上述步骤即可得到强学习器
$$
    f_M(x) = f_0(x) + \sum_{m=1}^M \sum_{j=1}^J \gamma_{jm}I(x\in R_{jm})
$$

### GBDT与提升树的关系
在回归问题中，loss函数为:
$$
    L(y, f(x)) = \frac{1}{2} (y - f(x))^2
$$
其负梯度方向为：
$$
    \frac{\partial L(y, f_{m-1}(x)}{\partial f_{m-1}(x)}= y - f_{m-1}(x)
$$
即为提升树中的残差项。**CART中叶子节点的值是训练集中被分配到该节点的样本的标签的均值：**
$$
    c_{jm} = \frac{1}{N_m}\sum_{i=1}^{N_m} y_i - f_{m-1}(x_i), \quad x_i \in R_{jm}
$$
公式(1)的最优解满足条件为梯度为0对应的点，即
$$
    \sum_{x_i \in R_{jm}}y_i - f_{m-1}(x_i) - \gamma_{jm} = 0
$$
因此：
$$
    \gamma_{jm} = \frac{1}{N_m} \sum_{x_i \in R_{jm}}y_i - f_{m-1}(x_i)
$$
结果和梯度提升树一致。

### GBDT正则化
1. 第一种是和Adaboost类似的正则化项，即步长(learning rate)。定义为$v$,对于前面的弱学习器的迭代
$$
    f_m(x) = f_{m-1}(x) + h_m(x)
$$
如果我们加上了正则化项，则有
$$
    f_m(x) = f_{m-1}(x) + vh_m(x)
$$
$v$的取值范围为$0\ltν\le1$。对于同样的训练集学习效果，较小的$v$意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。
2. 第二种正则化的方式是通过子采样比例（subsample）。取值为(0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在[0.5, 0.8]之间。
使用了子采样的GBDT有时也称作随机梯度提升树(Stochastic Gradient Boosting Tree, SGBT)。由于使用了子采样，程序可以通过采样分发到不同的任务去做boosting的迭代过程，最后形成新树，从而减少弱学习器难以并行学习的弱点。
3. 第三种是对于弱学习器即CART回归树进行正则化剪枝。在决策树原理篇里我们已经讲过，这里就不重复了。

