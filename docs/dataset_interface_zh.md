# MedEinst 数据集接口

[MedEinst 数据集](https://huggingface.co/datasets/zhui711/MedEinst)提供一个用于支持病例选择的**参考集（Reference Set）**和一个用于诊断评估的**基准测试集（Benchmark Test Set）**。ECR-Agent 还内置了一份固定的 853 条记录支持集清单，供离线 CGME 使用。

## 数据资源

| 资源 | 文件 | 描述 |
| --- | --- | --- |
| 参考集 | `train.jsonl` | 10,689 对反事实病例，用作训练资源。 |
| 基准测试集 | `test.jsonl` | 5,383 对反事实病例，涵盖 49 种疾病。 |
| 平衡种子子集 | [`data/medeinst_support_853.json`](../data/medeinst_support_853.json) | 为离线 CGME 选取的 853 条独立记录：涵盖 46 个疾病 ID，包含 522 条对照（control）与 331 条陷阱（trap）变体。 |

固定的支持集清单只存储病例身份标识与标签，不存储患者叙述文本。`run_cgme.py` 命令会按清单中的顺序，将这些记录与参考集进行匹配解析。

## 记录格式

每条 JSONL 记录包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `case_id` | 成对的对照（control）与陷阱（trap）变体共享的标识符 |
| `case_type` | `control` 或 `trap` |
| `narrative` | 提供给智能体的临床病例描述 |
| `ground_truth` | 相应变体的正确诊断 |

适配器同样接受可选的 `age` 和 `sex` 字段。对照与陷阱变体通过 `case_id` 配对，同时保持为相互独立的观察。疾病名称会被映射到相应的疾病 ID，临床叙述则按原样保留。

作为 JSONL 的替代方案，接口还接受 `case_*/final_benchmark_pair.json` 格式的成对文件目录。

## 使用公开数据集运行

使用固定支持集的离线 CGME：

```bash
python scripts/run_cgme.py \
  --reference-set /path/to/train.jsonl \
  --support-manifest data/medeinst_support_853.json
```

在线 ECR-Agent 推理：

```bash
python scripts/run_ecr_agent.py \
  --benchmark-set /path/to/test.jsonl \
  --refined-graph-dir /path/to/refined_graphs \
  --memory-bank-dir /path/to/memory_bank
```

在线命令还接受 `--limit N`、`--case-ids id1,id2`、`--skip-control` 与 `--skip-trap` 来选择病例或变体。离线命令支持串行与并发处理选项。

## 可选的支持集采样

若想构造一份新的候选支持集清单，而不是使用固定选择：

```bash
python scripts/prepare_support_set.py \
  --reference-set /path/to/train.jsonl \
  --output /path/to/candidate_support.json \
  --seed 42 --cap 20
```

该命令按病理设上限，对单个病例变体进行采样，并写出单独的候选清单。它不会取代固定的 `medeinst_support_853.json` 选择。

在线推理会生成 `results_detail.jsonl`、`inference_summary.json` 以及每个病例的图输出文件。基准评测指标在[论文](https://arxiv.org/abs/2601.06636)中定义。

关于模型配置与知识产物，请参阅 [配置](configuration.md) 与 [产物](artifacts.md)。
