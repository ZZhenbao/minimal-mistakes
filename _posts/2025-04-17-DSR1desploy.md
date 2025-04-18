---
title: "DeepSeek-R1(671B)部署方案"
date: 2025-04-18
categories: blog
---

# 模型信息
[nvidia/DeepSeek-R1-FP4 · Hugging Face](https://huggingface.co/nvidia/DeepSeek-R1-FP4)
1. **模型名称**: DeepSeek-R1（671B）
2. **隐藏层维度:** hidden_size: 7168
3. **Transformer 层数:** num_hidden_layers: 61
4. **注意力头数:** num_attention_heads: 128
5. **键值头数:** num_key_value_heads: 128
6. **FFN 中间层维度:**
    - 密集层 (Dense): intermediate_size: 18432
    - MoE 层: moe_intermediate_size: 2048
7. **MoE 配置:**
    - moe_layer_freq: 1 - 每个 Transformer 层都是 MoE 层。
    - n_routed_experts: 256 - 每个 MoE 层有 256 个可选专家。
    - n_shared_experts: 1 - 有 1 个共享专家（具体实现可能需要查看代码，通常 MoE 层会有一个共享的 FFN 或者所有专家共享某些参数）。
    - num_experts_per_tok: 8 - 每个 Token 在每个 MoE 层会路由到 8 个专家。 
8. **词汇表大小 (Vocabulary Size):** vocab_size: 129280
9. **最大序列长度 (Max Position Embeddings):** max_position_embeddings: 163840 - 模型理论上能处理的最大上下文长度。
10. **位置编码 (Positional Encoding):** 使用 RoPE (Rotary Positional Embedding) 及其变种 (rope_scaling type: "yarn")。
11. 激活函数: `silu`
12. **计算精度 (Torch Dtype):** torch_dtype: bfloat16 - 模型期望的计算精度是 BF16。
13. **量化方式**（假设部分）：
	- **FP4 权重量化**
	- 激活值仍使用 BF16
	- 可能为混合精度部署（MoE 中可能部分仍为 FP8）

# 推理显存消耗分析

## 核心假设

| 参数                    | 值                  |
| --------------------- | ------------------ |
| Batch Size            | `B = 32`           |
| 序列长度Sequence          | `S = 4096 / 16384` |
| Hidden Size           | `H = 7168`         |
| 层数                    | `L = 61`           |
| 参数精度                  | FP4（1 byte）        |
| KV Cache 精度           | BF16（2 bytes）      |
| 激活精度                  | BF16（2 bytes）      |
| KV Cache: key + value | ×2                 |

## 消耗分析
### **1. 模型参数:**
- 权重已使用 FP4 量化。
- **公式:** `Total_Physical_Parameters * Bytes_per_Parameter_FP4`
- **计算:** ~671B parameters * 0.5 byte/parameter = **335.5 GB**
- **说明:** 分布在所有参与 Tensor Parallelism 的 GPU 上。如果 tp_size = N，则每张 GPU 大约加载 335.5 / N GB 的权重。

### **2. KV Cache:**
- KV Cache 随 Batch Size 和 Sequence Length 线性增长。
- **公式:** `num_hidden_layers * B * S * 2 * hidden_size * Bytes_per_Element_KV`
    - num_hidden_layers: 61
    - B: 32
    - S: 序列长度 (4096 或 16384)
    - 2: 代表 Key 和 Value
    - hidden_size: 7168
    - Bytes_per_Element_KV: BF16 (2 bytes) 
- **计算 (BF16 KV Cache):**
    - S = 4096: 61 * 32 * 4096 * 2 * 7168 * 2 bytes ≈ **214.7 GB**
    - S = 16384: 61 * 32 * 16384 * 2 * 7168 * 2 bytes ≈ **858.9 GB**
- **说明:** KV Cache 通常会沿 Batch 或 Head 维度均匀分布到所有 TP GPU 上。

**3. 激活值 (Activations):**

| S     | Activation（估计） |
| ----- | -------------- |
| 4096  | ~60 - 80 GB    |
| 16384 | ~120 - 160 GB  |

 $Activation Memory=B×S×H×L×Bytes per Element$
- 难精确估算的部分，受具体实现、计算精度 (BF16) 和优化策略 (Activation Recomputation) 影响极大。

**4. 框架和中间缓冲区 (Framework Overhead):**
- 包括 CUDA 上下文、推理框架引擎、临时计算缓冲区等。随着 Batch Size 和 TP Size 的增加，这部分开销也可能略微增加。
- **预估值:** **5 - 10 GB** (总计，分布在各 GPU)

**总显存消耗估算汇总 (Total VRAM across all GPUs):**

| 场景                   | 序列长度 (S) | KV缓存精度         | 权重 (GB) | KV缓存 (GB) | 激活值 (GB) | 开销 (GB) | 总显存 (GB)     |
| -------------------- | -------- | -------------- | ------- | --------- | -------- | ------- | ------------ |
| **S=4096, BF16 KV**  | 4096     | BF16 (2 bytes) | ~336    | ~215      | ~80      | ~8      | **~639 GB**  |
| **S=16384, BF16 KV** | 16384    | BF16 (2 bytes) | ~336    | ~859      | ~150     | ~8      | **~1353 GB** |

**重要说明: 激活值高度不确定性**

# QLoRA微调显存消耗分析

## QLoRA 微调核心假设


| 参数 | 值| 说明  |
| ---------------------------------- | ----------------------------------------- | ------------------------------------------ |
| **Batch Size (B)**                 | 32                                        |                                            |
| **序列长度 (S)**                       | 4096 / 16384                              |                                            |
| **Hidden Size (H)**                | 7168                                      |                                            |
| **层数 (L)**                         | 61                                        |                                            |
| **LoRA Rank (r)**                  | 8 / 64                                    | LoRA 适配器的秩，直接影响可训练参数量 (选择两个典型值进行分析)        |
| **LoRA Target Modules**            | Attention Q/V Projections                 | 假设 LoRA 应用于注意力层中的 Query 和 Value 线性层 (常见做法) |
| **梯度精度**                           | BF16 (2 bytes)                            |                                            |
| **优化器 (Optimizer)**                | AdamW (Paged AdamW often used with QLoRA) |                                            |
| **优化器状态精度**                        | BF16 (2 bytes / state)                    |                                            |
| **KV Cache 精度**                    | BF16 (2 bytes)                            |                                            |
| **激活值精度 (Activations)**            | BF16 (2 bytes)                            |                                            |
| **梯度检查点 (Gradient Checkpointing)** | 启用 (Enabled)                              |                                            |
| **KV Cache: key + value**          | ×2                                        |                                            |
| **基础模型精度 (加载)**                    | 4-bit (0.5 bytes/param)                   | 使用 QLoRA 加载的基础模型参数精度                       |


## 消耗分析 (QLoRA 微调)

### 1. 基础模型权重 (Frozen, 4-bit Quantized Parameters)

- 使用 QLoRA 加载量化后的基础模型。
- **总物理参数量:** 671 B (671 * 10^9)
- **加载精度:** 4-bit (0.5 bytes/parameter)
- **计算:** 671 * 10^9 parameters * 0.5 bytes/parameter ≈ **335.5 GB*

### 2. LoRA 参数 (Trainable Parameters)
- 依赖于 hidden_size, num_layers, r，与基础模型总参数量无关。
- **计算 (BF16, 2 bytes):**
    - **r = 8:** ~13.99 M params => **28 MB**
    - **r = 64:** ~111.9 M params => **224 MB**  

### 3. 梯度 (Gradients)
- 只为 LoRA 参数存储梯度。
- **计算 (BF16, 2 bytes):**
    - **r = 8:** **28 MB**
    - **r = 64:** **224 MB**

### 4. 优化器状态 (Optimizer States)
- 通常使用 Paged AdamW 优化器配合 QLoRA，状态存储在 CPU RAM 或 NVMe，按需加载到 GPU。但为保守估计 VRAM，我们仍按标准 AdamW 状态暂存于 GPU VRAM 计算。
- **计算 (假设 BF16 状态, 4 bytes total/param):** `num_lora_params * 2 states * 2 bytes/state`
    - **r = 8:** 13.99 M * 4 bytes ≈ **56 MB**
    - **r = 64:** 111.9 M * 4 bytes ≈ **448 MB**
- **说明:** 若使用 Paged AdamW，这部分 VRAM 占用可能显著降低。

### 5. KV Cache
- 计算方式和结果不变，因为它取决于维度和序列长度，且通常使用 BF16 存储。
- **计算 (BF16 KV Cache):**
    - **S = 4096:** 61 * 32 * 4096 * 2 * 7168 * 2 bytes ≈ **214.7 GB**
    - **S = 16384:** 61 * 32 * 16384 * 2 * 7168 * 2 bytes ≈ **858.9 GB**

### 6. 激活值 (Activations)
- QLoRA 的前向/后向传播计算通常在 BF16 下进行，需要动态反量化基础模型权重参与计算。梯度检查点是必须的。
- **关键假设:** **启用梯度检查点**。
- **预估值 (BF16 计算精度, 启用梯度检查点):** 估算与之前的 LoRA 类似，但可能因反量化操作略有增加。
    - **S = 4096:** **~80 - 130 GB** (总计，分布在各 GPU)
    - **S = 16384:** **~150 - 270 GB** (总计，分布在各 GPU)
- **说明:** 依然是**高度不确定**的估算，强烈依赖梯度检查点策略。

### 7. 框架和中间缓冲区 (Framework Overhead)
- 包括 CUDA 上下文、QLoRA 实现（如 bitsandbytes）、训练框架、通信缓冲区等。可能比标准 LoRA 稍高。
- **预估值:** **10 - 20 GB** (总计，分布在各 GPU)

## 总显存消耗估算汇总 (QLoRA, 671B Base, 4-bit)


| 场景                   | 序列长度 (S) | LoRA秩 (r) | 基础模型 (GB, 4-bit) | LoRA参数+梯度+优化器 (GB) | KV缓存 (GB) | 激活值 (GB)        | 开销 (GB) | 总显存 (GB)     | 每GPU显存 (TP=8)   |
| -------------------- | -------- | --------- | ---------------- | ------------------ | --------- | --------------- | ------- | ------------ | --------------- |
| **A: S=4096, r=8**   | 4096     | 8         | ~335.5           | ~0.11 GB           | ~214.7    | ~105 (mid-est.) | ~15     | **~670 GB**  | **~84 GB/GPU**  |
| **B: S=4096, r=64**  | 4096     | 64        | ~335.5           | ~0.90 GB           | ~214.7    | ~105 (mid-est.) | ~15     | **~671 GB**  | **~84 GB/GPU**  |
| **C: S=16384, r=8**  | 16384    | 8         | ~335.5           | ~0.11 GB           | ~858.9    | ~210 (mid-est.) | ~15     | **~1420 GB** | **~178 GB/GPU** |
| **D: S=16384, r=64** | 16384    | 64        | ~335.5           | ~0.90 GB           | ~858.9    | ~210 (mid-est.) | ~15     | **~1421 GB** | **~178 GB/GPU** |

# 部署方案
部署DeepSeek-R1（671B），权重使用FP4量化，激活值和KV缓存使用BF16。
- **推理**：支持序列长度S=4096和16384，批大小B=32。
- **微调**：使用QLoRA，LoRA秩r=8和r=64，序列长度S=4096和16384，批大小B=32。
- **方法**：使用张量并行（TP）分布模型到多GPU。

### 推理显存消耗

| 场景               | 序列长度 (S) | KV缓存精度         | 权重 (GB) | KV缓存 (GB) | 激活值 (GB) | 开销 (GB) | 总显存 (GB)  |
| ---------------- | -------- | -------------- | ------- | --------- | -------- | ------- | --------- |
| S=4096, BF16 KV  | 4096     | BF16 (2 bytes) | ~336    | ~215      | ~80      | ~8      | **~639**  |
| S=16384, BF16 KV | 16384    | BF16 (2 bytes) | ~336    | ~859      | ~150     | ~8      | **~1353** |

### QLoRA微调显存消耗

| 场景            | 序列长度 (S) | LoRA秩 (r) | 基础模型 (GB, 4-bit) | LoRA参数+梯度+优化器 (GB) | KV缓存 (GB) | 激活值 (GB) | 开销 (GB) | 总显存 (GB)  | 每GPU显存 (TP=8) |
| ------------- | -------- | --------- | ---------------- | ------------------ | --------- | -------- | ------- | --------- | ------------- |
| S=4096, r=8   | 4096     | 8         | ~335.5           | ~0.11              | ~214.7    | ~105     | ~15     | **~670**  | ~84 GB        |
| S=4096, r=64  | 4096     | 64        | ~335.5           | ~0.90              | ~214.7    | ~105     | ~15     | **~671**  | ~84 GB        |
| S=16384, r=8  | 16384    | 8         | ~335.5           | ~0.11              | ~858.9    | ~210     | ~15     | **~1420** | ~178 GB       |
| S=16384, r=64 | 16384    | 64        | ~335.5           | ~0.90              | ~858.9    | ~210     | ~15     | **~1421** | ~178 GB       |

## 硬件配置

### 推理硬件

- **序列长度S=4096**：
    - **GPU**：8个NVIDIA A800 80GB GPU。
    - **并行策略**：张量并行（TP=8）。
    - **显存分配**：总需求639 GB，每GPU约79.875 GB（<80 GB）。
- **序列长度S=16384**：
    - **GPU**：17个NVIDIA A800 80GB GPU。
    - **并行策略**：张量并行（TP=17）。
    - **显存分配**：总需求1353 GB，每GPU约79.59 GB（<80 GB）。

### 微调硬件
- **序列长度S=4096，LoRA秩r=8或r=64**：
    - **GPU**：9个NVIDIA A800 80GB GPU。
    - **并行策略**：张量并行（TP=9）。
    - **显存分配**：总需求670-671 GB，每GPU约74.44-74.56 GB（<80 GB）。
- **序列长度S=16384，LoRA秩r=8或r=64**：
    - **GPU**：18个NVIDIA A800 80GB GPU。
    - **并行策略**：张量并行（TP=18）。
    - **显存分配**：总需求1420-1421 GB，每GPU约78.89 GB（<80 GB）。

# GPU信息

## **1. 高端计算卡（适合大规模 AI 训练/HPC）**

| GPU                     | 显存容量<br>（VRAM/HBM） | 显存带宽      | 计算性能 (稠密)<br>（TFLOPS/TOPS）                                                 | 支持精度                                        | 多卡互联带宽<br>（NVLink/NVSwitch）          | PCIe带宽             | 中国大陆市场价格 (RMB, 仅供参考) | 备注                       |
| ----------------------- | ------------------ | --------- | -------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------ | ------------------ | -------------------- | ------------------------ |
| **H100 80GB SXM5 (受限)** | 80GB HBM3          | 3.35 TB/s | FP64: 34<br>FP32: 67<br>TF32: 989<br>BF16/FP16: 989<br>INT8/FP8: 1979 TOPS | FP64, FP32, TF32, BF16, FP16, **FP8**, INT8 | 900 GB/s (NVLink 4.0 via NVSwitch)   | 128 GB/s (5.0 x16) | ¥250,000 - 350,000+  | 最新 Hopper 架构，性能巅峰 (出口受限) |
| **H100 80GB PCIe (受限)** | 80GB HBM3          | 2 TB/s    | FP64: 26<br>FP32: 51<br>TF32: 756<br>BF16/FP16: 756<br>INT8/FP8: 1513 TOPS | FP64, FP32, TF32, BF16, FP16, FP8, INT8     | 600 GB/s (NVLink 4.0 Bridge, 2卡)     | 128 GB/s (5.0 x16) | ¥240,000 - 330,000+  | Hopper 架构 PCIe 版本 (出口受限) |
| **A100 80GB SXM (受限)**  | 80GB HBM2e         | 2039 GB/s | FP64: 9.7<br>FP32: 19.5<br>TF32: 156<br>BF16/FP16: 312<br>INT8: 624 TOPS   | FP64, FP32, TF32, BF16, FP16, INT8          | 600 GB/s (NVLink 3.0 via NVSwitch)   | 64 GB/s (4.0 x16)  | ¥100,000 - 150,000+  | Ampere 旗舰 (原版，出口受限)      |
| **A800 80GB SXM (受限)**  | 80GB HBM2e         | 2039 GB/s | FP64: 9.7<br>FP32: 19.5<br>TF32: 156<br>BF16/FP16: 312<br>INT8: 624 TOPS   | FP64, FP32, TF32, **BF16**, FP16, INT8      | 400 GB/s (受限版 NVLink 3.0)            | 64 GB/s (4.0 x16)  | ¥120,000 - 180,000+  | A100 特供版 (现也受限)          |
| **A100 80GB PCIe (受限)** | 80GB HBM2e         | 1935 GB/s | FP64: 9.7<br>FP32: 19.5<br>TF32: 156<br>BF16/FP16: 312<br>INT8: 624 TOPS   | FP64, FP32, TF32, BF16, FP16, INT8          | **600 GB/s** (NVLink 3.0 Bridge, 2卡) | 64 GB/s (4.0 x16)  | ¥90,000 - 140,000+   | Ampere PCIe 版 (原版，出口受限)  |

#### **2. 中高端计算卡（适合 AI 推理/中小规模训练/图形）**

| GPU                    | 显存容量<br>（VRAM/HBM） | 显存带宽      | 计算性能 (稠密)<br>（TFLOPS/TOPS）                                          | 支持精度                              | 多卡互联带宽<br>（NVLink/NVSwitch）              | PCIe带宽            | 中国大陆市场价格 (RMB, 仅供参考)         | 备注                                 |
| ---------------------- | ------------------ | --------- | ------------------------------------------------------------------- | --------------------------------- | ---------------------------------------- | ----------------- | ---------------------------- | ---------------------------------- |
| **L40S (受限)**          | 48GB GDDR6         | 864 GB/s  | FP32: 91.6<br>TF32: 366<br>BF16/FP16: 366<br>INT8/FP8: 733 TOPS     | FP32, TF32, BF16, FP16, FP8, INT8 | N/A                                      | 64 GB/s (4.0 x16) | ¥90,000 - 130,000+ (稀缺/灰色市场) | 数据中心 GPU (AI/Graphics/HPC), (出口受限) |
| **RTX 6000 Ada**       | 48GB GDDR6         | 960 GB/s  | FP32: 91.1<br>TF32: 364.5<br>BF16/FP16: 364.5<br>INT8/FP8: 729 TOPS | FP32, TF32, BF16, FP16, FP8, INT8 | 100 GB/s (Ada NVLink Bridge, 2卡)         | 64 GB/s (4.0 x16) | ¥45,000 - 60,000             | 专业工作站 GPU (Ada 架构), 非消费级           |
| **RTX A6000 (Ampere)** | 48GB GDDR6         | 768 GB/s  | FP32: 38.7<br>TF32: 155<br>BF16/FP16: 155<br>INT8: 310 TOPS         | FP32, TF32, BF16, FP16, INT8      | 112.5 GB/s (NVLink 3.0 Bridge, 2卡)       | 64 GB/s (4.0 x16) | ¥30,000 - 40,000             | 专业工作站 GPU (Ampere 架构)              |
| **A40**                | 48GB GDDR6         | 696 GB/s  | FP32: 37.4<br>TF32: 149.7<br>BF16/FP16: 149.7<br>INT8: 299 TOPS     | FP32, TF32, BF16, FP16, INT8      | 112.5 GB/s (NVLink 3.0 Bridge, 2卡)       | 64 GB/s (4.0 x16) | ¥70,000 - 110,000+           | 数据中心推理/图形卡                         |
| **V100S PCIe**         | 32GB HBM2          | 1134 GB/s | FP64: 8.2<br>FP32: 16.4<br>FP16: 130<br>INT8: 130 TOPS              | FP64, FP32, FP16, INT8            | N/A (可选 NVLink 2.0 Bridge, 2卡: 150 GB/s) | 32 GB/s (3.0 x16) | ¥15,000 - 30,000             | Volta 架构增强版 PCIe 卡 (旧型号)           |
| **V100 SXM2**          | 32GB / 16GB HBM2   | 900 GB/s  | FP64: 7.8<br>FP32: 15.7<br>FP16: 125<br>INT8: 125 TOPS              | FP64, FP32, FP16, INT8            | 300 GB/s (NVLink 2.0)                    | 32 GB/s (3.0 x16) | ¥12,000 - 25,000             | Volta 架构 SXM2 接口卡 (旧型号)            |
| **V100 PCIe**          | 32GB / 16GB HBM2   | 900 GB/s  | FP64: 7<br>FP32: 14<br>FP16: 112<br>INT8: 112 TOPS                  | FP64, FP32, FP16, INT8            | N/A (可选 NVLink 2.0 Bridge, 2卡: 150 GB/s) | 32 GB/s (3.0 x16) | ¥10,000 - 20,000             | Volta 架构标准 PCIe 卡 (旧型号)            |

#### **3. 性价比推理卡（适合 AI 推理/边缘计算）**

| GPU            | 显存容量<br>（VRAM/HBM）   | 显存带宽              | 计算性能 (稠密)<br>（TFLOPS/TOPS）                                                    | 支持精度                                   | 多卡互联带宽<br>（NVLink/NVSwitch） | PCIe带宽            | 中国大陆市场价格 (RMB, 仅供参考) | 备注                          |
| -------------- | -------------------- | ----------------- | ----------------------------------------------------------------------------- | -------------------------------------- | --------------------------- | ----------------- | -------------------- | --------------------------- |
| **L4**         | 24GB GDDR6           | 300 GB/s          | FP32: 30.3<br>TF32: 121<br>BF16/FP16: 121<br>INT8/FP8: 242 TOPS               | FP32, TF32, BF16, FP16, FP8, INT8      | N/A                         | 32 GB/s (4.0 x8)  | ¥10,000 - 15,000     | 高效能推理 GPU (Ada 架构), PCIe x8 |
| **A10G (A10)** | 24GB GDDR6           | 600 GB/s          | FP32: 31.2<br>TF32: 125<br>BF16/FP16: 125<br>INT8: 250 TOPS                   | FP32, TF32, BF16, FP16, INT8           | N/A                         | 64 GB/s (4.0 x16) | ¥8,000 - 12,000      | 云服务器常用 (Ampere 架构)          |
| **A16**        | 64GB (4x 16GB) GDDR6 | 800 GB/s (4x 200) | (**卡总计**) <br>FP32: 21.6<br>TF32: 86.4<br>BF16/FP16: 86.4<br>INT8: 172.8 TOPS | FP32, TF32, BF16, FP16, INT8 (per GPU) | N/A                         | 64 GB/s (4.0 x16) | ¥20,000 - 30,000     | 4 GPU 合卡, VDI/推理优化 (Ampere) |
| **T4**         | 16GB GDDR6           | 320 GB/s          | FP32: 8.1<br>FP16: 65<br>INT8: 130 TOPS<br>INT4: 260 TOPS                     | FP32, FP16, INT8, INT4                 | N/A                         | 16 GB/s (3.0 x16) | ¥5,000 - 8,000       | 经典推理卡 (Turing 架构), PCIe 3.0 |

#### **4. 消费级 GPU（适合预算有限的小规模 AI/学习）**

|                    |                    |           |                                                                 |                                   |                                    |                   |                      |                              |
| ------------------ | ------------------ | --------- | --------------------------------------------------------------- | --------------------------------- | ---------------------------------- | ----------------- | -------------------- | ---------------------------- |
| GPU                | 显存容量<br>（VRAM/HBM） | 显存带宽      | 计算性能 (稠密)<br>（TFLOPS/TOPS）                                      | 支持精度                              | 多卡互联带宽<br>（NVLink/NVSwitch）        | PCIe带宽            | 中国大陆市场价格 (RMB, 仅供参考) | 备注                           |
| **RTX 4090 (受限)**  | 24GB GDDR6X        | 1008 GB/s | FP32: 82.6<br>TF32: 330<br>BF16/FP16: 330<br>INT8/FP8: 660 TOPS | FP32, TF32, BF16, FP16, FP8, INT8 | N/A                                | 64 GB/s (4.0 x16) | ¥15,000 - 20,000+    | 最强消费级 GPU (原版受限, 有 4090D 替代) |
| **RTX 4080 Super** | 16GB GDDR6X        | 736 GB/s  | FP32: 52.2<br>TF32: 209<br>BF16/FP16: 209<br>INT8/FP8: 418 TOPS | FP32, TF32, BF16, FP16, FP8, INT8 | N/A                                | 64 GB/s (4.0 x16) | ¥8,000 - 10,000      | 高性价比 AI 卡 (Ada 架构)           |
| **RTX 3090 Ti**    | 24GB GDDR6X        | 1008 GB/s | FP32: 40<br>TF32: 160<br>BF16/FP16: 160<br>INT8: 320 TOPS       | FP32, TF32, BF16, FP16, INT8      | 112.5 GB/s (NVLink 3.0 Bridge, 2卡) | 64 GB/s (4.0 x16) | ¥8,000 - 12,000      | 旧旗舰 (Ampere), 支持 NVLink      |
