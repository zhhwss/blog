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

## 用户行为模型

理解用户是搜索排序中一个非常重要的问题，工业级的推荐系统一般需要大量的泛化特征来较好的表达用户。这些泛化特征可以分为两类：
- 偏静态的特征，例如用户的基本属性（年龄、性别、职业等等）特征、长期偏好（品类、价格等等）特征；
- 动态变化的特征，例如刻画用户兴趣的实时行为序列特征。

用户的实时行为特征能够明显加强不同样本之间的区分度，所以在模型中优化用户行为序列建模是让模型更好理解用户的关键环节。
推荐系统中的用户兴趣变化非常剧烈，比如电商推荐中，用户一会看看服饰，一会看看电子产品，若只使用静态特征进行推荐，每次推荐的内容是一样的，这无疑是不能满足用户需求，实时性需要保障。
这里需要说明的是:
- 用户历史行为一般是指用户历史点击过商品或者广告
- 基于用户历史行为预测用户对当前商品或者广告的点击率或者转化率

### Pooling
第一种对用户历史行为建模的方式是使用Mean Pooling、Sum Pooling等Pooling操作按位聚合用户历史行为对应的Embedding Vectors，如下图所示,这里需要说明的是:
- 用户历史行为items的和当前item共享embedding矩阵
![](images/2021-08-09-22-11-19.png)

上述方法的主要问题是：
* 忽略了目标item与用户先前历史行为序列各item之间的相关性，导致模型无法从用户丰富的历史行为中捕捉用户多样的兴趣
事实是在一个用户浏览购买的过程中：
兴趣是多样的：
比如一个女性用户喜欢买女装，也喜欢化妆品，还可能同时还喜欢母婴用品，那么该女性用户在购买大衣的时候，要把对母婴用品的偏好考虑进来么？所以在对指定item预估CTR的时候，不能只是对历史Behavior进行简单pooling。

* 忽略了用户历史行为各个item背后的在时序性

### Attention
Pooling方法的主要问题之一，是忽略了目标item与用户先前历史行为序列各item之间的相关性，**换句话说是就是在做Pooling操作时每个历史item的权重是相等的。**
Attention机制NLP领域常用对hidden states向量求当前state下系数的方法。简而言之，Attention机制可以认为是一个小的神经网络，模拟
$$
    f(V_i, V_{current})\rightarrow\alpha_i
$$
的映射, $V_i$是历史某个item的embedding vector, $V_{current}$是当前item的embedding vector。用户历史的行为的特征最终表示为:
$$
    V_{history} = \sum_{i=1}^N f(V_i, V_{current})V_i
$$
需要注意的是，上式是按位逐元相加。如下图所示
![](images/2021-08-09-23-04-23.png)

其中典型的Attention网络有：
- 点乘
$$
     f(V_i, V_{current})=V_i^TV_{current}
$$
- 双线性
$$
    f(V_i, V_{current}) = V_i^T W  V_{current}
$$
- 缩放点积
$$
    f(V_i, V_{current})=\frac{V_i^TV_{current}}{\sqrt{d}}
$$
- 加性模型
$$
    f(V_i, V_{current})=W_1^Ttanh(W_2 V_i+ W_3 V_3)
$$
- 也可以构建一个小的MLP输入是$[V_i, V_{current}]$，输出是$V_i$的系数大小。

### DIN
DIN本质上是Attention方法的一种，其中构建了一个Activation Unit来学习
$$
    f(V_i, V_{current})\rightarrow\alpha_i
$$
![](images/2021-08-09-23-34-18.png)
其中Activation Unit本质上就是一个小的神经网络。
* 首先是把$V_i, V_{current}$，以及$V_i, V_{current}$的element wise差值向量和乘积向量concat起来作为输入
* 然后fed给全连接层
* 最后得出权重

值得一提的是DIN论文中提出了一种新的激活函数Dice
#### Dice
首先PReLU是一种常用的激活函数，其定义如下
$$
f(s)=\left\{\begin{array}{ll}
s & \text { if } s>0 \\
\alpha s & \text { if } s \leq 0
\end{array}=p(s) \cdot s+(1-p(s)) \cdot \alpha s\right.
$$
其中, $p(s)$是PReLu的控制函数，定义如下：
$$
p(s)=\left\{\begin{array}{ll}
1 & \text { if } s>0 \\
0 & \text { if } s \leq 0
\end{array}\right.
$$
Dice认为上述控制函数没有考虑数据的特征，提出如下图所示控制函数
![](images/2021-08-10-11-43-37.png)
其中，$E(s)是s的均值$，$p(s)$的定义如下：
$$
p(s)=\frac{1}{1+e^{-\frac{s-E[s]}{\sqrt{\operatorname{Var}[s]+\epsilon}}}}
$$
其中$E[s],Var[s]$是一个batch内输入数据的均值的方长。

### DIEN
#### DIN的不足

利用用户行为序列特征，直接把用户历史行为当做兴趣

直接用行为表示兴趣可能存在问题。因为行为是序列化产生的，行为之间存在依赖关系，比如当前时刻的兴趣往往直接导致了下一行为的发生。

用户的兴趣是不断进化的，而DIN抽取的用户兴趣之间是独立无关联的，没有捕获到兴趣的动态进化性，比如用户对衣服的喜好，会随季节、时尚风潮以及个人品味的变化而变化，呈现一种连续的变迁趋势。

为了解决上述问题，阿里进一步提出了DIEN网络，结构如下图所示：
其中主要结构是Interest Extractor Layer和Interest Evolution Layer
![](images/2021-08-10-15-01-37.png)

#### Interest Extractor Layer
兴趣抽取层Interest Extractor Layer的主要目标是从embedding数据中提取出interest。但一个用户在某一时间的interest不仅与当前的behavior有关，也与之前的behavior相关，所以作者们使用GRU单元来提取interest。GRU单元的表达式如下：
$$
\begin{aligned}
&\mathbf{u}_{t}=\sigma\left(W^{u} \mathbf{i}_{t}+U^{u} \mathbf{h}_{t-1}+\mathbf{b}^{u}\right) \\
&\mathbf{r}_{t}=\sigma\left(W^{r} \mathbf{i}_{t}+U^{r} \mathbf{h}_{t-1}+\mathbf{b}^{r}\right) \\
&\tilde{\mathbf{h}}_{t}=\tanh \left(W^{h} \mathbf{i}_{t}+\mathbf{r}_{t} \circ U^{h} \mathbf{h}_{t-1}+\mathbf{b}^{h}\right) \\
&\mathbf{h}_{t}=\left(\mathbf{1}-\mathbf{u}_{t}\right) \circ \mathbf{h}_{t-1}+\mathbf{u}_{t} \circ \tilde{\mathbf{h}}_{t}
\end{aligned}
$$
其中, $\mathbf{i}_{t}$表示GRU的输入，即历史第t个item的embedding 向量$\mathbf{e}(t)$， $\mathbf{h}_{t-1}$是GRU单元输入到下一步的隐状态，$\circ$表示按位乘, **这里我们可以认为$\mathbf{h}_{t}$是提取出的用户兴趣**.
为了训练好GRU和embedding layer，DIEN中引入了auxiliary loss如下图所示，包括：
- 当前item的embedding vector$\mathbf{e}(t+1)$ 和 GRU隐状态$\mathbf{h}_{h}$对应的正例
- 当前负采样item的embedding vector$\mathbf{e}(t+1)'$ 和 GRU隐状态$\mathbf{h}_{h}$对应的负例

aux loss形式为：
$$
\begin{aligned}
L_{a u x}=-& \frac{1}{N}\left(\sum_{i=1}^{N} \sum_{t} \log \sigma\left(\mathbf{h}_{t}, \mathbf{e}_{b}^{i}[t+1]\right)\right.\\
&\left.+\log \left(1-\sigma\left(\mathbf{h}_{t}, \hat{\mathbf{e}}_{b}^{i}[t+1]\right)\right)\right),
\end{aligned}
$$
其中$\mathbf{e}_{b}^{i}[t+1]$是当前item的embedding向量，$\hat{\mathbf{e}}_{b}^{i}[t+1]$是负采样的item的embedding向量。最终损失函数为$L=L_{target}+\alpha L_{aux}$ ，其中$\alpha$是平衡最终预测和兴趣表示的超参数。
![](images/2021-08-10-16-06-29.png)

#### Interest Evolution Layer
兴趣进化层Interest Evolution Layer的主要目标是刻画用户兴趣的进化过程。举个简单的例子：

以用户对衣服的interest为例，随着季节和时尚风潮的不断变化，用户的interest也会不断变化。这种变化会直接影响用户的点击决策。建模用户兴趣的进化过程有两方面的好处：
- 追踪用户的interest可以使我们学习final interest的表达时包含更多的历史信息。
- 可以根据interest的变化趋势更好地进行CTR预测。

而interest在变化过程中遵循如下规律：
- interest drift：用户在某一段时间的interest会有一定的集中性。比如用户可能在一段时间内不断买书，在另一段时间内不断买衣服。
- interest individual：一种interest有自己的发展趋势，不同种类的interest之间很少相互影响，例如买书和买衣服的interest基本

DISN中提出了将Attention加入到GRU来实现刻画用户兴趣的进化过程。
即在GRU中加入了Attention score。
首先是当前兴趣$\mathbf{h}_{t}$对应的Attention score计算公式为：
$$
a_{t}=\frac{\exp \left(\mathbf{h}_{t} W \mathbf{e}_{a}\right)}{\sum_{j=1}^{T} \exp \left(\mathbf{h}_{j} W \mathbf{e}_{a}\right)}
$$
其中$\mathbf{e}_{a}$是当前要预测的item的embedding 向量。
但是如何将attention机制加到GRU中呢？文中提出了三个方法：
* 直接改变GRU的输入(AIGRU):
这种方式将attention直接作用于输入，无需修改GRU的结构：
$$
     \mathbf{i}'_{t}=\alpha_t \mathbf{h}_{t}
$$
* Attention based GRU(AGRU)
这种方式需要修改GRU的结构，此时hidden state的输出变为：
$$
\mathbf{h}'_{t}=\left(\mathbf{1}-\alpha_t\right) \circ \mathbf{h}'_{t-1}+\alpha_t \circ \tilde{\mathbf{h}}_{t}
$$
* GRU with attentional update gate (AUGRU)
这种方式需要修改GRU的结构，此时hidden state的输出变为:
$$
\begin{aligned}
&\tilde{\mathbf{u}}'_{t}=\alpha_t \mathbf{u}'_t \\
&\mathbf{h}'_{t}=\left(\mathbf{1}-\tilde{\mathbf{u}}'_{t}\right) \circ \mathbf{h}'_{t-1}+\tilde{\mathbf{u}}'_{t} \circ \tilde{\mathbf{h}}'_{t}
\end{aligned}
$$
如下图所示:
![](images/2021-08-10-15-31-18.png)

### DSIN