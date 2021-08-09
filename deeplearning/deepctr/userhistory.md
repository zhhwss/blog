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
