---
{"dg-publish":true,"permalink":"/论文解读/模型分析/GLM-4.5 发布：面向推理、代码与智能体的开源SOTA模型/","title":"GLM-4.5 发布：面向推理、代码与智能体的开源SOTA模型","tags":["gardenEntry"]}
---


**摘要**  
>LLM 始终以跨广泛领域实现人类水平的认知能力为目标。在过去五年中，OpenAI的GPT-3学习了理解知识，o1使用强化学习先思考后响应，显著提高了编码、数据分析和复杂数学方面的推理能力。然而，这些模型仍然不是真正通用的：一些擅长编码，一些擅长数学，一些擅长推理，但它们都无法在所有不同的任务中实现最佳性能。GLM-4.5 努力实现统一所有不同功能的目标。
## 1. 引言

GLM 系列发布  两款新成员：**GLM-4.5** 和 **GLM-4.5-Air**，这两款模型在推理、编程和 Agentic 能力方面表现出色，代表了智谱在通用人工智能（AGI）之路上的重要里程碑。

- **GLM-4.5**：总参数 3550 亿，激活参数 320 亿
    
- **GLM-4.5-Air**：总参数 1060 亿，激活参数 120 亿
    

二者均具备：

- Thinking 模式：复杂推理和工具调用
    
- Non-thinking 模式：即时响应
    

提供平台：Z.ai, BigModel.cn  
开放权重：HuggingFace, ModelScope

---

## 2. 性能测试

在涵盖 agentic（3 项）、推理（7 项）、编程（2 项）共 12 项基准测试中，总体而言：

- **GLM-4.5 排名第3**
    
- **GLM-4.5-Air 排名第6**
    

![Pasted image 20250728223723.png](/img/user/Pasted%20image%2020250728223723.png)

---

### 2.1 Agentic 能力

GLM-4.5 拥有原生函数调用、128K 上下文长度，针对以下任务表现突出：
![Pasted image 20250728224017.png](/img/user/Pasted%20image%2020250728224017.png)
-  **τ-bench 和 BFCL-v3**上性能与 Claude 4 Sonnet 的性能相当
- **BrowseComp 得分 26.4%**，超越 Claude-Opus，接近 o4-mini-high（28.3%）
    

---

### 2.2 推理能力

在思维模式下，GLM-4.5和GLM-4.5-Air可以解决包括数学、科学和逻辑问题在内的复杂推理问题。

| Benchmark | GLM-4.5 |  o3  | Claude 4 Opus | Gemini 2.5 | Grok 4 |
| :-------: | :-----: | :--: | :-----------: | :--------: | :----: |
| MMLU Pro  |  84.6   | 85.3 |     87.3      |    86.2    |  86.6  |
|  AIME24   |  91.0   | 90.3 |     75.7      |    88.7    |  94.3  |
| MATH 500  |  98.2   | 99.2 |     98.2      |    96.7    |  99.0  |
|   GPQA    |  79.1   | 82.7 |     79.6      |    84.4    |  87.7  |

---

### 2.3 编程能力

GLM-4.5 不仅能编写完整项目，也能 agentic 地在已有代码中修复 Bug、添加功能。
![Pasted image 20250728224351.png](/img/user/Pasted%20image%2020250728224351.png)

|Benchmark|GLM-4.5|GPT-4.1|Claude 4 Opus|Claude 4 Sonnet|Kimi K2|
|---|---|---|---|---|---|
|SWE-bench Verified|64.2|48.6|67.8|70.4|65.4|
|Terminal-Bench|37.5|30.3|43.2|35.5|25.0|

此外：

- GLM-4.5 和 GLM-4.5-Air 相对于同等规模的型号表现出卓越的性能，在性能规模权衡边界上实现了最佳效率
    
- GLM-4.5 展示了全面的全栈开发功能，能够无缝创建包含前端实施、数据库管理和后端部署的 Web 应用程序
    
- GLM-4.5 工具调用成功率最高达 **90.6%**，优于 Claude-4-Sonnet（89.5%）、Kimi-K2（86.2%）和 Qwen3-Coder（77.1%）
    

---

## 3. Demo 展示


- **Flappy Bird 游戏**（HTML5 生成）

![Pasted image 20250728224852.png](/img/user/Pasted%20image%2020250728224852.png)
- **PPT & 海报生成代理**（自动搜索图文+排版）

![Pasted image 20250728225209.png](/img/user/Pasted%20image%2020250728225209.png)
![Pasted image 20250728225231.png](/img/user/Pasted%20image%2020250728225231.png)
- **Web 全栈项目开发**（如 Pokedex，宝可梦图鉴网站）
![Pasted image 20250728225406.png](/img/user/Pasted%20image%2020250728225406.png)

---

## 4. 架构与训练技术

### 4.1 模型架构与预训练

GLM-4.5 采用 MoE 架构，并在自注意力中引入 Grouped-Query Attention（GQA）+ 部分 RoPE 编码；头数达 96（hidden size 为 5120），提升推理表现。

- 减宽增深：加深层数、减小隐藏宽度 → 强化推理能力
    
- Muon 优化器：快速收敛，适配大 batch
    
- QK-Norm：稳定 attention logits
    
- MTP（Multi-Token Prediction）层：推理时支持 speculative decoding
    

![Pasted image 20250728230026.png](/img/user/Pasted%20image%2020250728230026.png)

- Pre-training：15T 通用语料 + 7T 代码推理语料
    
- Mid-training：500B Repo 级代码 + 500B 合成推理 + 100B 长上下文+Agent 数据
    


---

### 4.2 强化学习与 slime 平台
![Pasted image 20250728230328.png](/img/user/Pasted%20image%2020250728230328.png)

GLM-4.5 的 RL 后训练使用了自研平台 **slime**，具备以下优势：

- 异步训练架构：解耦 rollout 与训练，保持 GPU 饱和
    
- Agent 解耦设计：数据生成与模型训练完全分离，适配高复杂度任务，加速长尾agent task
    
- 混合精度：推理采用 FP8，训练用 BF16，吞吐提升
    

最终能力迁移至通用 tool-use、agentic 编程任务。

---
### 4.3 Agentic RL Post-training

后训练阶段采用**多专家 → 自蒸馏 → 统一强化学习**三阶段训练：

![Pasted image 20250728230938.png](/img/user/Pasted%20image%2020250728230938.png)

- 阶段一：分别针对推理、Agent、通用能力三个方面训练，冷启动+RL，训练出专家模型
    
- 阶段二：将三个专家蒸馏进base model当中
    
- 阶段三：进行一轮整体的SFT并依次对三个领域进行RL训练，得到最终的 GLM-4.5

整个后训练主要针对推理、Agent、通用能力三个场景，其中：

- **推理 RL**：单阶段 64K 上下文 + 课程学习 + 动态温度 + 自适应 clipping → 稳定优化 STEM 推理策略
    
- **Agentic RL**：基于信息检索 QA 与真实软件任务，利用 search-based QA 合成 & 代码执行反馈指导训练，这两个任务都有可验证的reward
    

尽管 RL 训练只针对少量具有可验证奖励的任务，但由此产生的收益会转移到相邻的能力，例如一般工具的使用。

___
## 5 开放使用方式

- 在线使用：[Z.ai](https://chat.z.ai/)
    
- API 接入：[BigModel.cn](https://bigmodel.cn/)
    
- 本地部署：HuggingFace & ModelScope 提供完整权重
    
- 框架支持：vLLM, SGLang, Claude Code 等
    
