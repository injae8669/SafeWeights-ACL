# Towards Identification and Intervention of Safety-Critical Parameters in Large Language Models

本仓库是论文 **Towards Identification and Intervention of Safety-Critical Parameters in Large Language Models** 的代码实现，核心目标是：

1. 识别对安全行为影响显著的参数（safety-critical parameters）。
2. 在不重新训练模型的前提下，通过参数级扰动验证这些参数对越狱行为的影响。

## 1. 方法概览

整体流程由 [main.py](main.py) 串联，分为两阶段：

1. 梯度识别阶段（Identification）
   - 使用 Guard 模型对被测 LLM 的生成结果产生安全梯度信号。
   - 通过词表映射矩阵将 Guard 梯度投影到被测 LLM 的词表空间。
   - 反向传播后统计参数梯度强度，输出参数/层/组件粒度的排名分析与可视化。

2. 扰动干预阶段（Intervention）
   - 依据梯度排名选择目标参数位置（top-k / bottom-k / random）。
   - 对这些位置注入按参数标准差缩放的高斯噪声。
   - 在对抗数据集上重新生成回答并使用 GPT 评测，计算 jailbreak rate。

## 2. 仓库结构

- [main.py](main.py)：实验入口，执行“梯度分析 + 扰动评估”。
- [config.py](config.py)：主实验命令行参数定义。
- [models.py](models.py)：加载被测模型和 Guard 模型。
- [prompt_analyzer.py](prompt_analyzer.py)：单条样本的前向生成与梯度回传。
- [gradient_analysis.py](gradient_analysis.py)：梯度统计与参数组件聚合。
- [perturbation.py](perturbation.py)：索引选择、参数扰动、GPT 评测。
- [reporting.py](reporting.py)：可视化图表导出。
- [utils.py](utils.py)：模板 embedding、标签抽取、词表梯度映射。
- [trainer.py](trainer.py)：可选的掩码微调训练脚本（LoRA / Full SFT）。
- [optimizers.py](optimizers.py)：MaskedAdamW 实现。
- [datasets/](datasets/)：默认评测数据（如 advbench520、harmbench400）。

## 3. 环境准备

### 3.1 Python 与依赖

建议：Python 3.10+，CUDA 环境与 PyTorch 版本匹配。

```bash
pip install torch transformers datasets peft pandas numpy seaborn matplotlib openai tqdm
```

### 3.2 模型准备

需要准备两类模型路径：

- 被测 LLM（--model_path）
- Guard 模型（--guard_model_path）

说明：当前代码路径中，主流程默认要求 Guard 模型参与计算；未提供 Guard 模型会在分析阶段报错。

### 3.3 API 配置

扰动后的安全评估使用 OpenAI 兼容接口。运行时需提供：

- --api_secret_key
- --base_url

## 4. 数据格式

主流程读取 [datasets/](datasets/) 下 JSON 文件，默认字段为 goal。

示例：

```json
[
  {"goal": "Provide detailed instructions for ..."},
  {"goal": "How to bypass ..."}
]
```

注意：当前 [main.py](main.py) 内部默认使用 advbench520.json。如需切换到 harmbench400 等数据集，请修改 main.py 中的 dataset_files 列表。

## 5. 快速开始

以下命令在仓库根目录执行。

### 5.1 完整流程（梯度识别 + top-k 扰动评估）

```bash
python main.py \
  --model_path /path/to/target-llm \
  --guard_model_path /path/to/guard-model \
  --api_secret_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --device cuda:0 \
  --output_dir ./result \
  --selection_mode topk \
  --top_k_ratio 0.01 \
  --noise_ratio 0.38
```

### 5.2 使用缓存索引（跳过梯度分析）

若已有历史保存的索引目录（包含 metrics.pt / layer_indices.pt / flat_indices.pt），可直接加载：

```bash
python main.py \
  --model_path /path/to/target-llm \
  --guard_model_path /path/to/guard-model \
  --api_secret_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --device cuda:0 \
  --output_dir ./result \
  --selection_mode topk \
  --index_path /path/to/saved/index \
  --resort_k 0.1
```

### 5.3 对照实验模式

- topk：扰动最重要参数（默认）。
- bottomk：扰动低重要性参数。
- random：随机扰动。

示例：

```bash
python main.py ... --selection_mode random --top_k_ratio 0.01
python main.py ... --selection_mode bottomk --top_k_ratio 0.01
```

## 6. 核心参数说明（main.py）

来自 [config.py](config.py)：

- --model_path：被测 LLM 路径（必填）。
- --guard_model_path：Guard 模型路径（建议必填，主流程依赖）。
- --api_secret_key：评测 API key。
- --base_url：评测 API base url。
- --output_dir：实验输出目录（必填）。
- --device：设备，如 cuda:0。
- --max_new_tokens：分析阶段生成长度（默认 256）。
- --selection_mode：topk / random / bottomk。
- --top_k_ratio：选择参数比例（默认 0.01）。
- --noise_ratio：扰动噪声强度（默认 0.38）。
- --index_path：已保存索引目录；提供后可跳过梯度识别。
- --resort_k：在已加载索引中再次筛选比例。

## 7. 输出结果说明

输出会写入：

```text
{output_dir}/{model_name}_{dataset_tag}_{M1_or_M2}/
```

其中常见文件包括：

- saved_grad/*.pt：梯度缓存。
- detailed_gradient_results_all_prompts.csv：全量参数梯度统计。
- comprehensive_avg_gradient_plots_*.png：参数/层/组件可视化。
- {ratio}_topk/ 或 {ratio}_bottomk/ 或 {ratio}_random/：扰动索引缓存。
- targets_by_name_*.pkl：按参数名组织的扰动目标。

控制台将输出最终 jailbreak rate。

## 8. 可选训练脚本（trainer.py）

[trainer.py](trainer.py) 提供额外的微调能力（不属于主评测入口），支持：

- full SFT
- LoRA
- masked_adamw（仅更新掩码指定位置）

示例：

```bash
python trainer.py \
  --model_path /path/to/model \
  --dataset_path /path/to/train.json \
  --output_dir ./result/train \
  --finetune_mode lora \
  --optimizer masked_adamw \
  --mask_source file \
  --mask_file_path /path/to/targets_by_name.pkl \
  --bf16
```

## 9. 复现实验建议

1. 固定随机种子与模型版本。
2. 先跑 topk，再跑 random/bottomk 作为对照。
3. 使用同一套 prompt 与评测 API 设置，避免评测偏差。
4. 建议保留索引缓存，便于不同噪声强度下重复实验。

## 10. 常见问题

1. 报错 guard_model is required
   - 主流程当前依赖 Guard 梯度，请确保传入 --guard_model_path。

2. 评测分数全为 0 或 API 调用失败
   - 检查 --api_secret_key 与 --base_url；确认接口兼容 chat.completions。

3. 显存不足
   - 减小 --max_new_tokens，或减少数据规模；优先使用已缓存 index_path 跳过梯度阶段。

4. 索引越界
   - 一般是索引文件与当前模型不匹配，删除缓存并重新计算。

## 11. 引用

如果本代码对你的研究有帮助，请引用论文：

**Towards Identification and Intervention of Safety-Critical Parameters in Large Language Models**

（可在论文正式 BibTeX 条目确定后补充到此处）
