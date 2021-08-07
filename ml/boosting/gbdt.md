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

到这里，就会引出两个问题：
- **对于分类问题，该如何构建子树**，使用残差项效率肯定非常低
- **如果单纯最小化残差项导致另一个问题，即很容易过拟合**

解决上述两个问题的办法是我在boosting的构建子树的时候，**目标不是拟合残差，而是使得loss函数最小**（loss函数里的正则项可以缓解过拟合问题）。

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
\gamma_{j m}=\arg \min _{\gamma} \sum_{x_{i} \in R_{j m}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\gamma\right), \quad j \in \{1,\cdots, J\}
$$
- 更新学习器
$$
    f_{m}(x) = f_{m-1} = \sum_{j=1}^J \gamma_{jm}I(x\in R_{jm})
$$
- 重复上述步骤即可得到强学习器
$$
    f_M(x) = f_0(x) + \sum_{m=1}^M \sum_{j=1}^J \gamma_{jm}I(x\in R_{jm})
$$
