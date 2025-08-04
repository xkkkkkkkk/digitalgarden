---
{"dg-publish":true,"permalink":"/论文解读/算法解读/Agentic Reinforced Policy Optimization (ARPO)：   面向多轮工具调用的Agentic RL新范式/","tags":["gardenEntry"]}
---



---
> **摘要**  
> ARPO针对**工具调用后LLM生成的不确定性激增**（表现为token熵显著上升）的现象提出了一种面向多轮工具调用场景的强化学习算法，其核心创新包括：  
> 1. **基于熵的自适应Rollout机制**：动态平衡全局轨迹采样与高熵步骤的分支采样，提升工具使用行为的探索效率，能够在更广泛的domain上采样agentic轨迹；  
> 2. **优势估计**：通过两种优势分配策略，使LLM内化工具调用步骤的奖励差异。  
>
> 实验表明，ARPO在13个数学推理、知识推理和深度搜索任务上超越SOTA方法，**仅需50%的工具调用预算**即可达到更高性能，为LLM智能体与动态环境对齐提供高效解决方案。

---

## 1. 引言：问题背景与研究思路
![Pasted image 20250730184442.png](/img/user/Pasted%20image%2020250730184442.png)
- 💡 **核心问题**：  
  论文发现**工具调用后LLM输出的token会出现熵突变现象**（如左图所示），也就是不确定性很强。而现在常见的一些RL方法（如GRPO）在这类长轨迹采样上并没有注意这一点，导致无法有效探索工具调用后的高不确定性区域
- 📈 **研究思路**：  
  1. 针对高熵区域进行多次采样，充分拓展了采样轨迹的多样性，这一点在数据合成上也是一个很不错的思路；  
  2. 针对这种采样方案设计了相对应的优势估计方案

---

## 2. 问题描述：如何定义Agentic RL
在深入探讨 Agentic Reinforced Policy Optimization (ARPO) 算法之前，理解其核心概念和背景至关重要。本节将对关键概念进行概述，并回顾基于熵的 LLM 推理实验
### 2.1 Agentic 强化学习（Agentic RL）

Agentic RL 的训练目标被制定为一个最大化问题：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta (:x; T)} [r_\psi(x, y)] - \beta D_{KL} [\pi_\theta(y|x; T)||\pi_{\text{ref}}(y|x; T)]
$$
其中

- $T$：可用的工具集
- $\pi_\theta$：策略 LLM（大型语言模型）
- $\pi_{\text{ref}}$：参考 LLM
- $r_\psi$：奖励函数
- $D_{KL}$：KL 散度
- $x$：从数据集 $D$ 中抽取的输入
- $y$：对应的输出

与传统 RL 方法不同，Agentic RL 在推理过程中融入了工具调用反馈，例如`compile_lit_review`，
`read_pdf`，这些工具能够帮助 LLM 在复杂任务中进行多轮交互

Rollout 采样可以表示为：

$$
P_\theta(R, y|x; T) = \left[ \prod_{t=1}^{t_R} P_\theta(R_t|R_{<t}, x; T) \right] \cdot \left[ \prod_{t=1}^{t_Y} P_\theta(Y_t|Y_{<t}, R, x; T) \right]
$$

其中：
- $R$：长度为 $t_R$ 的推理轨迹
- $y$：长度为 $t_Y$ 的最终答案

ARPO 基于规则的 RL 算法（如 GRPO）旨在优化基于 LLM 的智能体（LLM-based agents）。

### 2.2 Agentic 推理中的 Token 熵分析

为了深入理解 LLM 在 Agentic 推理中的行为，需要对 Token 熵进行分析，根据最近基于熵的 RL 研究，Token 级别生成熵 $H_t$ 在步骤 $t$ 计算如下：

$$
H_t = -\sum_{j=1}^V p_{t,j} \log p_{t,j}, \quad \text{where } p_t = \text{Softmax} \left( \frac{z_t}{\tau} \right)
$$
其中
- $V$：词汇表大小。
- $z_t \in \mathbb{R}^V$：pre-softmax logits。
- $\tau$：解码温度。
![Pasted image 20250730192210.png](/img/user/Pasted%20image%2020250730192210.png)
这个熵值反映了 Token 生成分布中的不确定性，而非特定 Token 的不确定性。通过对 LLM-based tool-use agents 的初步研究，观察到：

- **观察 1 (Ob.1)**：每次工具调用反馈后，LLM 生成的初始 10-50 个 Token 的熵值急剧上升，表明外部工具调用显著增加了 LLM 推理过程中的不确定性。
- **观察 2 (Ob.2)**：熵值在早期推理阶段倾向于增加，但仍低于接收工具调用反馈后的水平，这揭示了 LLM-based agents 尚未充分探索的潜在行为。
- **观察 3 (Ob.3)**：搜索反馈比 Python 反馈引入更多不确定性。这是因为搜索引擎通常返回信息丰富的文本内容，而 Python 输出包含确定性数字，导致前者的熵波动更大。

这些发现揭示了传统 trajectory-level RL 方法的局限性，它们侧重于初始推理，却忽略了工具调用反馈引入的不确定性。ARPO 算法通过整合基于熵的探索机制来解决这一问题。

### 2.3 Agentic 工具设计

论文选定了三种代表性工具来评估 ARPO 的有效性：

- **搜索引擎（Search Engine）**：通过执行网络查询来检索相关信息。
- **网页浏览器代理（Web Browser Agent）**：访问并解析搜索引擎返回的相关网页链接，提取并总结关键内容。
- **代码解释器（Code Interpreter）**：自动执行语言模型生成的代码，成功则返回执行结果，否则返回编译器错误消息。

这些初步概念和工具构成了 ARPO 算法的基础，旨在通过理解和利用 LLM 在工具交互中的不确定性来优化其学习过程。


## 3. 解决方案：熵导向的Agentic RL框架

### 3.1 整体设计思路  
![Pasted image 20250730192457.png](/img/user/Pasted%20image%2020250730192457.png)
ARPO将强化学习过程解耦为两个协同模块：  
1. **熵监测器**：实时量化工具调用后的token熵变化 $\Delta H_t$；  
2. **自适应rollout引擎**：基于$\Delta H_t$动态触发分支采样，平衡全局推理与局部探索（图3）。

### 3.2 核心技术创新：熵导向自适应Rollout
**步骤流程**（图4左）：  
1. **熵初始化**：生成$N$条全局轨迹，计算初始$k$个token的熵$H_{initial}$；  
2. **熵监测**：每轮工具调用后，生成$k$个token并计算当前熵$H_t$，归一化两者的差：  
   $$\Delta H_t = \text{Normalize}(H_t - H_{initial})$$  
3. **分支决策**：当采样概率$P_t = \alpha + \beta \cdot \Delta H_t > \tau$时，也就是说不确定性超过阈值后，就从当前节点分支$Z$条子轨迹：  
   $$\text{Action}(P_t) = \begin{cases} 
   \text{Branch}(Z) & \text{if } P_t > \tau \\
   \text{Continue} & \text{otherwise}
   \end{cases}$$
其中 $α$ 为基本采样概率（base sampling probability）, $β$ 为熵变权重系数
![Pasted image 20250730193343.png](/img/user/Pasted%20image%2020250730193343.png) 
4. **终止条件**：总采样数达$M$或所有路径终止。  

### 3.3 适配分支采样的优势估计
在这种分支采样的方法下，应该用一种更公平的方案来衡量轨迹中独立部分与重合部分，因此提出两种优势分配策略：  
- **硬优势估计**：**明确地、分割地**计算并分配每一步操作的优势。对于共同的操作序列，它们会获得一个**合并的平均优势** $\hat{A}_{\text{shared}}$；对于不同的操作序列，它们会获得**独立的优势**$\hat{A}_{\text{ind}}$。
- **软优势估计**：通过GRPO目标函数隐式地来分配：  
  $$J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1\pm\epsilon) \hat{A}_{i,t} \right) \right]$$  
软优势估计的核心思想为：不强行把轨迹分割成共享和非共享部分来计算单独的优势，而是让模型在**整体的优化框架下**，通过调整生成不同轨迹的概率（通过重要性采样比率），来**隐式地**学习哪些步骤在哪些上下文中最有价值。这意味着，即使某一个工具调用动作是共享的，它在每个轨迹中的“影响”或“权重”也会因为后续不同的步骤和最终奖励而有所不同，而不是简单地取平均

### 3.4 奖励模型设计
ARPO 的奖励函数设计灵感借鉴了 **Tool-Star**，考虑**正确性**和**格式**的奖励机制，并在此基础上加入了**多工具协作奖励**。ARPO 在此基础上进行了优化，其总奖励 RR 的定义如下：

$$
R = 
\begin{cases} 
\max(Acc. + r_M, Acc.) & \text{If Format is Good } \& Acc.>0 \\ 
0 & \text{If Format is Good } \& Acc.=0 \\ 
-1 & \text{Otherwise}
\end{cases}
$$

$$
r_M =
\begin{cases} 
0.1 & \text{If } \exists (\text{<search>} \& \text{<python>}) \\ 
0 & \text{Otherwise}
\end{cases}
$$
### 3.5 技术优势与创新点
- **动态探索**：首次将token熵作为探索信号，替代人工设计奖励函数；  
- **理论保障**：提出广义策略梯度定理（GPG Theorem），证明分段优化有效性：  
  $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left\{ \sum_{T=1}^{K} \left[ \nabla_\theta \log \pi_\theta(MA_T | MS_T) \Phi_T \right] \right\}$$  
- **工程友好**：减少50%工具调用次数，显著降低训练成本。

---

## 4. 实验评估与结果分析
### 4.1 实验设计
- **数据集**：13个基准任务，涵盖三类场景：  
  - *数学推理*：AIME24/25, MATH500, GSM8K  
  - *知识推理*：HotpotQA, Musique, WebWalker  
  - *深度搜索*：GAIA, HLE, xBench  
- **评估指标**：Pass@1准确率，工具调用次数，熵分布统计；  
- **基线对比**：GRPO, DAPO, REINFORCE++，以及Search-o1、ReAct等工作流方法；  
- **训练设置**：Qwen2.5/Llama3系列模型，仅用1K RL样本微调。

### 4.2 核心结果
数学推理与知识推理的结果为：
![Pasted image 20250730194726.png](/img/user/Pasted%20image%2020250730194726.png)
Deep search的结果为
![Pasted image 20250730194747.png](/img/user/Pasted%20image%2020250730194747.png)
**关键结论**：  
1. **效率优势**：ARPO仅用50%工具调用次数即超越GRPO；  
2. **深度搜索优化**：在GAIA任务中Pass@5达61.2%，证明分支采样提升行为多样性；  
3. **熵阈值敏感度**：$\Delta H_t$权重$\beta=0.4$时性能最优，过高导致探索失衡。

---

## 5. 核心贡献与技术价值
### 5.1 主要贡献总结
- **理论贡献**：  
  - 首次量化工具调用引发的LLM熵分布变化；  
  - 提出GPG定理支持分段强化学习。  
- **技术贡献**：  
  - 熵导向rollout机制动态优化探索效率；  
  - 优势归因估计适配分支采样结构。  
- **工程贡献**：  
  - 工具调用次数减少50%，降低部署成本；  
  - 开源代码库提供完整训练框架。

### 5.2 对领域的影响
- **推动Agentic RL发展**：为多轮交互任务建立新的训练范式；  
- **启发研究方向**：熵信号可作为通用探索指标扩展至其他RL场景；  
- **工业价值**：为搜索增强RAG、多工具协作系统提供高效对齐方案。

---

### 5. 局限性与未来方向
#### 5.1 当前局限
- **长程依赖挑战**：超过10轮工具调用时分支路径爆炸；  
- **静态熵阈值**：固定$\tau$可能不适应复杂任务动态；  
- **浏览器代理依赖**：强依赖外部工具质量（如表3）。

#### 5.2 改进方向
- **熵预测模块**：用轻量模型预测最优分支时机；  
- **课程学习**：从简单任务逐步提升工具调用轮次；  
- **工具感知RL**：联合优化工具选择与调用策略。
