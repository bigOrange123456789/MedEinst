<div align="center">

# MedEinst

**通过反事实鉴别诊断对医学大语言模型中的“定势效应”（Einstellung Effect）进行基准评测**

<p>
  <a href="https://arxiv.org/abs/2601.06636"><img src="https://img.shields.io/badge/Paper-arXiv%3A2601.06636-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper" /></a>
  <a href="https://zhui711.github.io/MedEinst/"><img src="https://img.shields.io/badge/Project-Website-2563EB?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Website" /></a>
  <a href="https://huggingface.co/datasets/zhui711/MedEinst"><img src="https://img.shields.io/badge/Benchmark-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="MedEinst Dataset" /></a>
</p>
<p>
  <a href="https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs"><img src="https://img.shields.io/badge/Illness-Graphs-0E7490?style=flat-square" alt="Illness Graphs" /></a>
  <a href="https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base"><img src="https://img.shields.io/badge/Exemplar-Base-7C3AED?style=flat-square" alt="Exemplar Base" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-64748B?style=flat-square" alt="Apache-2.0 License" /></a>
</p>

</div>

## 概览

大语言模型可能会偏向于熟悉的诊断结论，即使患者特异的证据指向其他方向——这正是**定势效应（Einstellung Effect）**的一种表现。**MedEinst** 通过反事实鉴别诊断来研究这一失效模式，并提出了 **ECR-Agent（基于证据的因果推理智能体，Evidence-based Causal Reasoning Agent）**，以支持更具证据依据的诊断推理。

本仓库提供 **MedEinst 基准评测资源以及主要方法 ECR-Agent 的源代码**。

## 🧪 MedEinst 基准与数据集

MedEinst 的基准测试集包含 **5,383 对反事实临床病例，涵盖 49 种疾病**。每一对病例都包含一个*对照*病例和一个*陷阱*病例：区分性证据改变了正确诊断，而大量临床背景仍然相似。论文对 **17 个大语言模型**进行了评测，并使用**基线准确率（Baseline Accuracy）、稳健准确率（Robust Accuracy）和偏差陷阱率（Bias Trap Rate）**来衡量诊断准确率以及模型对该偏差的易感性。

| 资源 | 内容 | 用途 |
| --- | --- | --- |
| **参考集**（`train.jsonl`） | 10,689 对病例 | 训练资源及支持病例的选择 |
| **基准测试集**（`test.jsonl`） | 5,383 对病例 · 49 种疾病 | 评估诊断稳健性与定势效应 |
| **平衡种子子集**（`data/medeinst_support_853.json`） | 853 条独立病例记录 | 用于离线 CGME 的固定支持集选择 |

**数据集：** [🤗 Hugging Face 上的 MedEinst](https://huggingface.co/datasets/zhui711/MedEinst) · [数据集接口说明](docs/dataset_interface.md)

临床叙述文本源自 [DDXPlus](https://github.com/mila-iqia/ddxplus) 并经过处理。我们感谢 DDXPlus 作者公开其数据集。

## 🩺 ECR-Agent

为缓解定势效应，论文提出了 **ECR-Agent**，它由两个互补的组件构成：

**动态因果推理（Dynamic Causal Inference，DCI）。** 执行双通路感知、因果图初始化、前向证据检索、反向验证以及证据审计，从而得出诊断结论。

**批评者驱动的图与记忆演化（Critic-Driven Graph & Memory Evolution，CGME）。** 利用支持病例和批评者反馈来演化特定疾病的**疾病图（Illness Graphs）**和**范例库（Exemplar Base）**。离线 CGME 从实时推理中构建这些知识；在线 DCI 则从预构建的知识出发进行初始化，并继续进行基于证据的推理。

请参阅[方法架构](docs/architecture.md)和[项目主页](https://zhui711.github.io/MedEinst/)以获取框架示意图。

## ⚙️ 安装与配置

推荐使用 Python 3.10 及以上版本。安装仓库依赖：

```bash
python -m pip install -r requirements.txt
```

在 `config/settings.yaml` 中分别配置基座 LLM（`api`）、批评者（`critical_model`）和嵌入服务（`embedding`）。LLM 支持直接加载本地权重，以及远程或本地 OpenAI 兼容 HTTP 服务。远程凭证推荐通过 `api_key_env` 指定的环境变量提供，也支持 `api_key`。

### 本地权重推理（与 lab2.py 的 localhost 模式一致）

在运行本项目的 Python 环境中安装可选依赖（例如先 `conda activate MedEinst`）：

```bash
python -m pip install -r requirements-local.txt
```

将 `settings.yaml` 的 `api` 段中的对应字段改为：

```yaml
api:
  localhost: true
  model_name: my-local-model       # 本地模式仅用于日志，实际加载 model_path
  model_path: 'G:/models/Qwen3-8B'  # 改成真实的完整模型目录
  lora_adapter: null               # 有 LoRA 时填写适配器目录
  torch_dtype: auto
  trust_remote_code: true
  max_new_tokens: 4096
  temperature: 0.0
  top_p: 0.95
  repetition_penalty: 1.0
  chat_template_kwargs:
    enable_thinking: false        # 仅对支持此模板参数的模型生效

embedding:
  enabled: false                  # 没有嵌入服务时关闭
```

这是字段修改示例，其余已有配置可以保留。`localhost: true` 时无需 API Key，`base_url` 和对应环境变量不会用于模型调用。`model_path` 必须是包含模型配置、tokenizer 和权重的 Transformers 目录；GGUF 文件请通过本地兼容 HTTP 服务使用。路径相对 `settings.yaml` 所在目录解析，Windows 路径建议使用 `/` 或 YAML 单引号。

设备由程序自动检测，无需在 YAML 中配置：优先使用当前 PyTorch 环境可用的 CUDA 显卡（`cuda:0`），其次使用 Apple MPS；没有可用显卡时使用 CPU。首次加载模型时会打印所选设备，加载完成后打印当前 GPU/CPU 运行模式，CUDA 模式还会显示显卡名称。判断依据是 PyTorch 能否使用显卡，因此 CPU 版 PyTorch 或不可用的驱动会选择 CPU。

运行离线 CGME 或图精化时，将 `critical_model` 段同样设置为 `localhost: true` 并填写模型目录；它与 `api` 相互独立。相同加载配置会复用一份权重，本地并发请求会排队生成。模型首次调用时才加载；加载失败会明确返回错误。普通 CausalLM 和 Qwen3-VL 等视觉语言模型均走文本推理，本项目没有增加图片输入流程。Qwen3-VL 需要支持该架构的 Transformers 版本（至少 4.57）。

`max_new_tokens`、`temperature` 设置后会覆盖各推理阶段传入的对应值，不设置则沿用阶段默认值；`chat_template_kwargs` 传给本地聊天模板。模板处理遵循 [Transformers 官方聊天模板说明](https://huggingface.co/docs/transformers/main/en/chat_templating)，LoRA 通过 [PEFT 的加载接口](https://huggingface.co/docs/peft/en/package_reference/peft_model)加载。

### 本地 OpenAI 兼容服务

已通过 Ollama、vLLM 或 LM Studio 启动服务时，无需本地权重依赖，设置例如：

```yaml
api:
  localhost: false
  base_url: http://127.0.0.1:8000/v1
  base_url_env: null              # 避免已有环境变量覆盖本地 URL
  api_key_env: null
  api_key: local                 # 无认证时任意非空值；需要认证时填真实密钥
  model_name: your-served-model  # 服务实际暴露的模型名
  supports_logprobs: false
```

loopback 地址（localhost / 127.0.0.1 / ::1）未提供密钥时自动使用占位值；其他地址仍要求配置密钥。服务专用参数通过 `extra_body` 透传。

嵌入服务独立于 LLM：`embedding.enabled: false` 会停用向量检索并使用已有文本匹配能力，因此不等同于完整的原始评测配置，也不会访问默认远程嵌入 API。若需要保留向量检索，可将 `embedding.base_url`、`model_name`、`dimension` 配成自己的兼容嵌入服务。更换嵌入模型后应使用新模型重建范例库向量，即使维度相同也不能混用。诊断流程中的外部知识搜索仍可能访问网络；本地模型模式不代表整个系统离线运行。

## 🚀 快速开始：在线 ECR-Agent

下载基准测试集（`test.jsonl`）、[疾病图（Illness Graphs）](https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs)和[范例库（Exemplar Base）](https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base)。解压这些预构建产物并配置好模型服务后，运行：

```bash
python scripts/run_ecr_agent.py \
  --benchmark-set /path/to/test.jsonl \
  --refined-graph-dir /path/to/golden_graphs_refined_1 \
  --memory-bank-dir /path/to/memory_bank \
  --output-dir /path/to/online_results
```

运行器会将病例级别的预测写入 `results_detail.jsonl`，并将运行摘要写入 `inference_summary.json`。关于预期的文件与目录结构，请参阅[产物布局](docs/artifacts.md)。三个基准指标在论文中已给出定义；本次发布的智能体代码中不包含单独的评估工具。

控制台默认以英文输出；添加 `--console-language zh` 可切换为中文，`--console-language en` 切回英文。查看中文帮助：`python scripts/run_ecr_agent.py --console-language zh --help`。Windows CMD 示例见[控制台语言配置](docs/configuration_zh.md#控制台语言)。

## 🔄 离线 CGME：构建知识与记忆

仓库自带的 `data/medeinst_support_853.json` 保留了论文中固定的支持集选择。从[参考集](https://huggingface.co/datasets/zhui711/MedEinst)下载 `train.jsonl`，然后运行：

```bash
python scripts/run_cgme.py \
  --reference-set /path/to/train.jsonl \
  --support-manifest data/medeinst_support_853.json \
  --output-dir /path/to/cgme_output
```

（可选）对生成的疾病图进行精化：

```bash
python scripts/refine_illness_graphs.py \
  --input-dir /path/to/cgme_output/golden_graphs \
  --output-dir /path/to/refined_graphs \
  --log-dir /path/to/refinement_logs
```

最终生成的疾病图和范例库可用于在线推理。更多细节请参阅[数据集接口](docs/dataset_interface.md)和[产物布局](docs/artifacts.md)。

## 📖 引用

如果你在研究中使用 MedEinst 或 ECR-Agent，请引用我们的论文：

```bibtex
@misc{chen2026medeinstbenchmarkingeinstellungeffect,
  title={MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis},
  author={Wenting Chen and Zhongrui Zhu and Guolin Huang and Wenxuan Wang},
  year={2026},
  eprint={2601.06636},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.06636}
}
```

## 📬 联系方式

**Zhongrui Zhu** — [zhongruizhuzz@outlook.com](mailto:zhongruizhuzz@outlook.com)
