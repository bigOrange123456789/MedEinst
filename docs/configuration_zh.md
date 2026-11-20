# ECR-Agent 配置

模型、API 与输出设置定义在 [`config/settings.yaml`](../config/settings.yaml) 中。CGME、在线推理以及疾病图精化命令默认加载该文件；可传入 `--config /path/to/settings.yaml` 使用其他配置文件。

凭证信息通过环境变量提供，而不是存储在 YAML 文件中。

## 模型端点

| YAML 配置节 | 默认模型 | 凭证环境变量 | 可选端点覆盖 |
| --- | --- | --- | --- |
| `api` | `qwen3-32b` | `ECR_BASE_API_KEY` | `ECR_BASE_URL` |
| `critical_model` | `gpt-4o` | `OPENAI_API_KEY` | `ECR_CRITIC_URL` |
| `embedding` | `text-embedding-3-small`（1536 维） | `OPENAI_API_KEY` | `ECR_EMBEDDING_URL` |

每个配置节都支持 `base_url` 与 `api_key_env` 设置。环境变量中的端点覆盖优先于该配置节的 `base_url`。若未配置自定义 URL，则使用 OpenAI API 端点。

默认的 `qwen3-32b` 模型需要兼容的模型服务端点。例如，当使用单独托管的基座模型，并用 OpenAI 处理批评者与嵌入请求时：

```bash
export ECR_BASE_URL=https://your-compatible-endpoint.example/v1
export ECR_BASE_API_KEY=your-base-model-key
export OPENAI_API_KEY=your-openai-key
```

请调整模型 ID 与端点以匹配可用的服务。论文使用 GPT-5 作为批评者模型；示例配置默认使用 `gpt-4o`，可通过 `critical_model.model_name` 进行修改。

本地 `.env` 文件也可通过 `python-dotenv` 加载；该文件不应提交到仓库。

## 数据与输出路径

YAML 中的相对路径相对于 YAML 文件本身解析。CLI 中的相对路径相对于当前工作目录解析；同时也支持绝对路径。

| 用途 | 配置 / CLI |
| --- | --- |
| CGME 的参考集 | `run_cgme.py --reference-set` |
| 固定支持集清单 | `run_cgme.py --support-manifest`（默认 `data/medeinst_support_853.json`） |
| 基准测试集 | `run_ecr_agent.py --benchmark-set` |
| 原始疾病图 | `online_inference.golden_graph_dir` / `--raw-graph-dir` |
| 精化疾病图 | `online_inference.refined_graph_dir` / `--refined-graph-dir` |
| 范例库 | `online_inference.memory_bank_dir` / `--memory-bank-dir` |
| 在线结果 | `online_inference.output_dir` / `--output-dir` |
| 离线 CGME 输出 | `offline_training.golden_graph_dir`、`memory_bank_dir`、`log_dir`；或 `run_cgme.py --output-dir` |

图精化支持 `--input-dir`、`--output-dir` 和 `--log-dir`。

## 控制台语言

`scripts/run_ecr_agent.py` 支持 `--console-language {en,zh}`：`en` 以英文为主（默认），`zh` 以中文为主。该参数控制启动信息、各推理阶段日志、检索与记忆库提示、进度条以及最终统计，并对并行工作线程生效。

Windows CMD 中使用中文输出：

```bat
python scripts/run_ecr_agent.py ^
  --benchmark-set "G:\Codex\MedEinst_dataset\test.jsonl" ^
  --output-dir "G:\Codex\MedEinst\output_smoke" ^
  --limit 1 ^
  --console-language zh
```

切换为英文时，将最后一行改为 `--console-language en`，也可省略该参数。CMD 中每个续行符 `^` 必须位于行尾，其后不要加空格。

查看中文参数帮助：

```bat
python scripts/run_ecr_agent.py --console-language zh --help
```

该选项仅改变控制台显示语言，不翻译模型输入、疾病名称、模型回复、第三方异常原文或保存的 JSON 字段。Python 异常堆栈也保留原文，方便排查问题。Windows 下脚本将标准输出和错误输出设为 UTF-8；重定向保存的日志也使用 UTF-8。

## 外部知识源

PubMed 检索使用 `PUBMED_EMAIL` 以及可选的 `PUBMED_API_KEY`。OpenTargets 检索使用其已配置的公共服务端点。

关于数据格式与产物目录，请参阅 [数据集接口](dataset_interface.md) 与 [产物](artifacts.md)。
