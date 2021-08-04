## DeepCTR
在DeepCTR模型通常应用于表数据场景，该场景通常由离散特征和连续特征组成。如下图所示，每一行表示一个样本，每一列表示特征。机器学习任务就是要根据特征x预测目标y。其中离散特征一般用one-hot encoding处理，连续特征用归一化处理。然而在实际应用中，例如CTR预测场景，离散特征可能包含了上百万的不同值，这也就导致了one-hot vector是高纬稀疏的，进而特征向量x也是高纬稀疏的。
![](images/2021-08-04-11-33-25.png)
**应用深度神经网络到该领域首先要解决的就是离散变量高维稀疏问题**。

### Embedding Vector
Deep模型能在CTR领域应用的最重要技术之一就是Embedding，Embedding技术可以将高纬稀疏的离散特征映射成低纬的稠密向量，如下图所示：
![](images/2021-08-04-15-32-06.png)
其定义如下 ：
$$
    \mathbf{e}_i =  \mathbf{W}_i \mathbf{x}_i \tag{1}
$$
其中，$\mathbf{x}_i\in R^n$是one-hot向量（在多值field时为multi-hot向量）, $\mathbf{W}_i\in R^{n\times d}$是embedding矩阵, $\mathbf{e}_i\in R^d$是embedding向量，n是field $i$中不同值的个数, d是embedding factor。实际上，公式(1)也可以被看作是取出矩阵$\mathbf{W}_i$的第$x_i$列。

此时，可以将离散变量的embedding向量和连续变量concat起来作为上层DNN的输入。即
$$
    \mathbf{X}_{dnn} = concat([\mathbf{e}_i,\cdots, x_j\cdots])
$$
其中$x_j$是连续特征。下图所示就是一个典型的DNN模型。
<image src="./images/2021-08-04-15-52-29.png" width=400>

### Wide & Deep
Wide & Deep 是Google 2016年提出的用于表数据的经典模型，该模型将DNN学习到的高阶特征表示、原始特征和专家构造的特征拼到一起，直接预测最终结果，如下图所示：
<image src="images/2021-08-04-15-58-45.png" width=500>
其定义如下：
$$
    P(Y=1|X)=\sigma(W_{wide}^T[X, \phi(X)] + W_{deep}^TDNN(X) + b)
$$
其中, $\phi(X)$是专家构造的特征, $DNN(X)$是DNN学习到的高阶特征表示，$X$是原始特征。
这里，Wide部分和Deep部分可以用不同的优化器来联合训练。例如，Wide部分用FTRL，Deep部分用adam。
目前主流的DeepCTR模型都是Wide & Deep模式，即用一个FM、CIN等网络替换Wide部分。

### DeepFM
DeepFM是将FM模块和Deep模块并联到一起，如下图所示：
![](images/2021-08-04-16-55-14.png)
FM是用来表征Field之间内积的网络表征方式，定义如下:
$$
    y_{FM}= w_0 + \mathbf{w_1}^T \mathbf{X} + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} <\mathbf{e_i}, \mathbf{e_j}>
$$
其中, $\mathbf{X}=concat([\mathbf{x_i}\cdots x_j\cdots])$是离散特征one-hot和连续特征拼接的结果，$e_i$和$e_j$是embedding 向量。
这里，很容易发现FM不能学到连续特征之间以及连续特征与离散特征之间的交互。一个简单的办法是将连续特征离散化后使用DeepFM，或者可通过一个映射矩阵，将连续特征映射到同embedding 向量相同的维度，如下：
$$
    \mathbf{e_j} = x_j \mathbf{V}_j
$$
其中，$\mathbf{V}_j$即为映射向量。
DeepFM的最终预测为:
$$
    \hat{y}=\sigma(y_{DNN} + y_{FM})
$$