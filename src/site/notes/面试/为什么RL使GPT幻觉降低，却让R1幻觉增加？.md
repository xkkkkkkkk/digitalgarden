---
{"dg-publish":true,"permalink":"/面试/为什么RL使GPT幻觉降低，却让R1幻觉增加？/","title":"为什么RL使GPT幻觉降低，却让R1幻觉增加？","tags":["gardenEntry"]}
---


---

## 1. RL对幻觉的影响，真的有差异吗？

### 1.1 GPT系列（OpenAI）与DeepSeek R1的幻觉率对比
>[!note] HHEM提供了一个LLM幻觉测试榜单中，幻觉率被定义为产生幻觉的回复占它生成的所有回复的比率，包含事实冲突，无中生有，指令误解，逻辑错误。

- **OpenAI**：
  - 在2022年提出的InstructGPT，也就是俗称的GPT3.5，和当时的ChatGPT有着相同的训练方式，都采用了RLHF的方式对模型进行训练，典型的RLHF对LLM的微调训练包含三个阶段：SFT提高指令遵循能力、训练reward model建模人类偏好、强化学习对齐人类偏好
  - 通过RLHF训练的的GPT3.5有着很低的幻觉率，之后每一代新模型的迭代，比如GPT-4,4o，通常都会在减少幻觉方面有所进步
  - 但其实在推理模型o3刚发布的时候，同时也出现了幻觉率极高的情况，前OpenAI研究员Neil Chowdhury表示，o系列模型使用的强化学习算法，可能是问题的根源
![Pasted image 20250720122604.png](/img/user/Pasted%20image%2020250720122604.png)
- **DeepSeek R1**：
  - 类似的问题也同样发生在deepseek r1上，r1是在v3的基础上经过两轮SFT＋RL训练的，比起v3幻觉率上升了很多，当然在后面发布的版本也逐步降低
![Pasted image 20250720120255.png](/img/user/Pasted%20image%2020250720120255.png)

- **结论**：确实存在“OpenAI RL早期训练后幻觉降低，DeepSeek R1 RL后幻觉升高”的现象。或者更严谨地说，通过rl训练出来的推理模型会有比基础模型更高的幻觉，而基于rlhf训练的一些模型反而幻觉更低

---

## 2. 原因分析：RL训练的相同与不同

### 2.1 大语言模型的幻觉

大语言模型的幻觉是很普遍的，主要来源于：
  1. 预训练的文本当中本来就或多或少存在误解、不确定性或可能性极低的事件，这使得模型天然有可能输出错误信息
  2. RL训练会使模型更加迎合用户的输入
  3. 在幻觉测试的场景下的训练数据分布与预训练不同
  4. Rollout阶段引入了随机性

### 2.2 关键差异点

#### 1. **Reward Model设计与数据分布**

- **大模型 RLHF**：
  - RM训练数据覆盖面广，标注标准严格，**对事实性和幻觉有明确惩罚**
  - RM设计上对齐人类偏好，幻觉高的response会被惩罚或者奖励很低

- **推理模型的RL后训练**：
  - RM训练数据可能更偏向流畅性、相关性等主观指标
  - RM设计上对齐代码，数学等结果，存在hacking的情况

#### 2. **KL惩罚项设置**

- **大模型 RLHF**：
  - PPO reward中KL项较大，防止policy偏离SFT太远，**抑制reward hacking**
  - 公式：  
$$
R(x, y) = r_\theta(x, y) - \beta \log \left[ \frac{\pi^{\text{RL}}_\phi(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right]
$$

- **推理模型的RL后训练**：
  - KL系数可能设置较小，RL优化步数较多，policy更容易“hack” reward model，导致幻觉率上升

#### 3. **RL训练细节与超参数**

- RL步数、batch size、reward normalization等细节不同，影响RLHF效果
- RL训练过度（overoptimization）会导致policy过拟合reward model的缺陷，幻觉反而增加


---

## 3. 如何降低幻觉？我们能学到什么？

### 3.1 技术层面

- **Reward Model设计**：
  - 将“事实性/幻觉”作为reward model的指标，收集高质量、覆盖多样幻觉类型的标注数据
  - 结合多目标reward（如factuality+helpfulness+harmlessness），提升模型整体可靠性

- **KL惩罚与正则化**：
  - 适当加大KL系数，防止policy偏离SFT太远，抑制reward hacking
  - 采用reward shaping、归一化等技术，防止RL训练过度

- **引入RAG**：
  - 大模型通常存在知识边界，检索增强生成（RAG）通过在生成过程中引入外部知识源（如数据库、文档或网页），使模型能够访问和利用最新的、相关的信息，从而提高回答的准确性

### 3.2 启发与建议

- **RL训练的框架是类似的，对幻觉的不同影响主要来源于reward model设计**，这两种reward的设计方法都不是完美的，其实本质还是“我们不知道自己的偏好到底是什么”
- **大模型的幻觉仍然是很严重的问题，一方面需要降低幻觉，一方面需要提升评测能力**

---

## 4. 参考资料

- [Learning to summarize from human feedback (OpenAI)](https://arxiv.org/pdf/2009.01325.pdf)
- [HaluEval-Wild: Evaluating Hallucinations of Language Models in the Wild](https://arxiv.org/html/2403.04307v3)
- [ODIN: Disentangled Reward Mitigates Hacking in RLHF](https://arxiv.org/abs/2402.07319)
- [Reward Shaping to Mitigate Reward Hacking in RLHF](https://arxiv.org/abs/2502.18770)
- [AI Models with the Lowest Hallucination Rates: A Revealing Analysis](https://opentools.ai/news/ai-models-with-the-lowest-hallucination-rates-a-revealing-analysis)
- [Medium: Hallucination in AI Text Generation](https://medium.com/@amanatulla1606/overview-252e604a659d)
