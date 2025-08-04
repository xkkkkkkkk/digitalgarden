---
{"dg-publish":true,"permalink":"/论文解读/算法解读/Qwen GSPO 解析：Sequence-level RL Algorithm/","title":"Group Sequence Policy Optimization (GSPO) 解析","tags":["RL","LLM","强化学习","PPO","GRPO","GSPO","gardenEntry"]}
---

---

## 1. 背景与动机

> **问题**：传统的基于 PPO 或 GRPO 的 RL 训练，往往使用 _token-level_ 的重要性采样比与裁剪，  
> 但**奖励信号**往往只在**序列级**给出，二者“**粒度不匹配**”导致方差积累和训练不稳定。

### 1.1 研究动机
随着模型规模的增长，特别是在稀疏结构（如 Mixture-of-Experts，MoE）中，为了最大化硬件利用率，RL 训练中通常使用**超大批量的 rollout 数据**。为了提高训练效率，这些 rollout 会被分割成多个 mini-batch 进行优化
- 这是一个典型的**off-policy**训练，数据来自旧策略 $\pi_{\theta_{\text{old}}}$ 而不是当前策略 $\pi_\theta$
    
- PPO 和 GRPO 引入“裁剪机制”是为了缓解这种 off-policy 带来的误差
    
- 然而这种机制只解决了表层问题，** GRPO 的优化目标在本质上是“病态”的 **，这种病态问题在训练大模型、处理长响应时尤其严重，可能会导致训练崩溃

>**本质原因：重要性采样（importance sampling）错用**
>
在理论上，importance sampling 的基本原理是：若我们希望从目标分布 $\pi_{\text{tar}}$ 估计一个函数 $f(z)$ 的期望，可以从行为分布 $\pi_{\text{beh}}$ 采样，然后用加权修正：
>$$
\mathbb{E}_{z \sim \pi_{\text{tar}}}[f(z)] = \mathbb{E}_{z \sim \pi_{\text{beh}}}\left[ \frac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)} f(z) \right] $$
这一原理前提是对行为分布进行**多样本平均**（$N \gg 1$），才能准确估计修正因子。
>而GRPO 对每个 token 使用的比率是：
>$$\frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} | x, y_{i,<t})} $$​
>
>也就是说，**每一个 token 的采样修正只基于单一样本 $y_i$**，这导致：
>
>- 无法正确修正分布偏移
>    
>- 引入高方差噪声，尤其在长序列中不断累积
>    
>- 配合 token-level 裁剪后，变本加厉
>
因此，如果奖励是对“完整序列”赋予的，那么进行策略更新时也应该在序列级别进行采样修正与优化，由此引进了一种全新算法**GSPO：Group Sequence Policy Optimization**

回顾一下PPO和GRPO
### 1.2 PPO（Proximal Policy Optimization）  
PPO（Schulman 等，2017）使用旧策略 $\pi_{\theta_{\text{old}}}$ 生成的样本，在策略优化过程中通过“裁剪机制”将策略更新限制在旧策略附近的一个邻域内。PPO 的优化目标如下）：

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\text{old}}(\cdot|x)}\left[
\frac{1}{|y|} \sum_{t=1}^{|y|} \min\left( w_t(\theta) \hat{A}_t,\; \mathrm{clip}(w_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)
\right],
$$
其中：

- $w_t(\theta)$ 是 token $y_t$ 的重要性比（importance ratio）：
$$w_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{old}}(y_t | x, y_{<t})}$$​
- $\hat{A}_t$ 是 $y_t$ 的优势值，本质就是在衡量当前状态下选取的动作是相比较与所有可选动作的平均水平是高还是低。PPO中采用广义优势估计GAE来估计优势值，该估算需要状态价值函数，而状态价值函数就是价值模型在近似的值，因此刚需价值模型，也就是critic model
    
- $\epsilon$ 是重要性比裁剪范围

PPO 应用在LLM微调上的最大难点在于其依赖critic model来近似状态价值函数
- 价值网络通常与策略网络规模相当，带来了显著的内存和计算负担
    
- 整体效果高度依赖于优势估计的可靠性

### 1.3 GRPO（Group Relative Policy Optimization）  
GRPO（Shao 等，2024）则**不再依赖价值模型**，而是通过对同一 query 下多个候选响应之间进行比较，本质上就是通过多次采样来判断当前选择的动作（也就是输出哪个token）是优于平均水平还是差于平均水平

其优化目标如下：
$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\text{old}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left(w_{i,t}(\theta) \hat{A}_{i,t},\; \mathrm{clip}(w_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) \right], 
$$
其中：

- $G$ 是针对每个输入 $x$ 生成的候选响应数量；
    
- 比率和优势分别为：
    $$
    w_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} | x, y_{i,<t})} , 
    \hat{A}_{i,t} = \hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_j)\}_{j=1}^G)}{\text{std}(\{r(x, y_j)\}_{j=1}^G)} $$
在 $y_i$ 中的所有 token都共享相同的优势 $\hat{A}_i$


---

## 2 核心算法

### 2.1 序列级重要性采样

定义序列级比率  
$$
s_i(\theta)
= \exp\Bigl(\tfrac1{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}\Bigr)
= \Bigl(\prod_{t=1}^{|y_i|}r_{i,t}\Bigr)^{1/|y_i|}
$$
本质上是将动作的尺度从token-level改变为sequence-level
- **优势**：
    
    1. 粒度匹配——按序列计算比率，与序列级奖励对齐。
        
    2. 方差控制——对数求和再指数，避免长序列上噪声放大。
        

### 2.2 序列级裁剪与优势估计

- **群内优势**
$$    
    A_i = \frac{r(x,y_i) - \tfrac1G\sum_{j=1}^G r(x,y_j)} {\sqrt{\tfrac1G\sum_{j=1}^G\bigl(r(x,y_j)-\bar r\bigr)^2}}
$$
- **GSPO 优化目标**
    $$
    J(\theta) = \mathbb{E}_{x,\{y_i\}\sim\pi_{\mathrm{old}}} \Bigl[\tfrac1G\sum_{i=1}^G \min\bigl(s_i(\theta)\,\hat A_i,\;\mathrm{clip}(s_i(\theta),1-\epsilon,1+\epsilon)\,\hat A_i\bigr) \Bigr]
    $$
GSPO 将裁剪应用于整个响应而不是单个标记，以从梯度估计中排除过度“偏离策略”的样本，这与序列级奖励和优化相匹配，由于重要性比的不同定义，GSPO 和以前的算法（例如 GRPO）中的裁剪范围通常在数量级上有所不同。
    

### 2.3 GSPO‑token 变体

在多轮对话或需要逐 token 调整场景中，GSPO-token 固定 $s_i(\theta)$，对每个 token 使用自定义优势 $\hat A_{i,t}$：
$$
J_{\text{token}}(\theta) = \mathbb{E}\Bigl[\sum_{t=1}^{|y_i|} \min\bigl(s_i(\theta)\,\hat A_{i,t},\; \mathrm{clip}(s_i(\theta),1-\epsilon,1+\epsilon)\,\hat A_{i,t}\bigr)\Bigr]
$$
- 当 $\hat A_{i,t}\equiv \hat A_i$ 时，退化为序列版 GSPO。
    
关于梯度的分析以及更详细的理论推导请见原论文

---

## 3 实验设置与结果

### 3.1 基准与指标

- **模型**：Qwen3-30B-A3B-Base、MoE-64-Experts
    
- **任务集**：AIME’24（数学推理）、LiveCodeBench（代码理解）、CodeForces（编程题）
    
- **对比方法**：GRPO（含/不含 Routing Replay）
    

### 3.2 收敛曲线对比

![Pasted image 20250727181331.png](/img/user/Pasted%20image%2020250727181331.png)


> 在相同训练步数下，GSPO 的**总奖励提升，波动显著降低**。

### 3.3 裁剪比例消融

- **GRPO 是对 token 层面裁剪**：每个 token 的重要性比 $r_{t}$ 都要检查是否超出了阈值
    
- **GSPO 是对整个序列裁剪**：一旦整条序列的重要性比超出了阈值，整条序列都被梯度估计时被忽略
![Pasted image 20250727182354.png](/img/user/Pasted%20image%2020250727182354.png)
如图在相同条件下，GSPO 剪掉的 token 数量比 GRPO **高出两个数量级**，**但令人惊讶的是**：即使 GSPO 使用的 token 数量**远远少于 GRPO**，它的训练效率**更高**，这说明
> 1 **留下的训练样本提供了更有价值、更稳定的梯度信号**，
> 2 **GRPO 的 token-level 梯度估计非常嘈杂、低效，不能有效利用样本**

### 3.4 MoE 模型稳定性分析
![Pasted image 20250727181732.png](/img/user/Pasted%20image%2020250727181732.png)
- **无 Routing Replay**：
    
    - GRPO 严重发散
        
    - GSPO 正常收敛，释放专家全部容量
        
- **结论**：序列级裁剪对路由波动有天然缓冲作用，无需额外机制。
    

---

## 4 参考文献

1. Zheng, C., et al. **Group Sequence Policy Optimization**. arXiv:2507.18071v1, 2025.