# DDIM 数学推导

## 1. 正向扩散（Forward Process）

DDIM 和 DDPM 使用相同的正向扩散过程：

$$
q(x_t | x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I\big), \quad t=1,...,T
$$

累积噪声的封闭形式：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中：

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

---

## 2. 从 \(x_t\) 预测 \(x_0\)

已训练好的噪声预测网络 \(\epsilon_\theta(x_t, t)\)：

$$
\epsilon \approx \epsilon_\theta(x_t, t)
$$

可以解出 \(x_0\) 的估计：

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

---

## 3. 非马尔可夫逆过程（DDIM Update）

由 \(\hat{x}_0\) 计算 \(x_{t-1}\)：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

- 当 \(\sigma_t = 0\) 时，采样完全确定性。
- 若 \(\sigma_t > 0\)，可加入少量随机性。

---

## 4. DDIM 采样算法

1. 初始化 \(x_T \sim \mathcal{N}(0,I)\)
2. 对 \(t=T,...,1\)：
   - 预测噪声 \(\epsilon_\theta(x_t, t)\)
   - 估计 \(x_0\)：
     $$
     \hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
     $$
   - 更新 \(x_{t-1}\)：
     $$
     x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t) + \sigma_t z
     $$
3. 输出最终生成样本 \(x_0\)

---

## 5. 与 DDPM 的区别

- DDPM：逐步马尔可夫随机采样，每步加噪声
- DDIM：非马尔可夫，可确定采样或少量随机，速度快，可逆
