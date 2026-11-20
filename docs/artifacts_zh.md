# 疾病图与范例库

ECR-Agent 可以使用随 MedEinst 提供的预训练产物：

- [疾病图（Illness Graphs）](https://huggingface.co/datasets/zhui711/MedEinst-Illness-Graphs)
- [范例库（Exemplar Base）](https://huggingface.co/datasets/zhui711/MedEinst-Exemplar-Base)

请从源码仓库中单独下载并解压这些资源。配置时应指向包含解压后文件的目录，而不是压缩包路径。

## 产物布局

```text
artifacts/
├── golden_graphs_refined_1/
│   └── *_Refined_GG.json
└── memory_bank/
    ├── memory_bank.json
    └── embeddings.npy
```

已发布的疾病图集合包含 **46 个精化疾病图**。将 `--refined-graph-dir` 设置为包含 `*_Refined_GG.json` 文件的目录。如果存在原始图，`--raw-graph-dir` 可指向包含 `*_GG.json` 文件的目录。

范例库包含 **700 条病例记录**以及形状为 **(700, 1536)** 的嵌入向量数组。将 `--memory-bank-dir` 设置为同时包含 `memory_bank.json` 和 `embeddings.npy` 的目录。查询所用的嵌入模型应与已存储的嵌入向量相兼容；仅维度匹配并不能保证嵌入语义一致。

## 在线推理

在配置好模型端点并下载基准数据集后，示例命令如下：

```bash
python scripts/run_ecr_agent.py \
  --benchmark-set /path/to/test.jsonl \
  --refined-graph-dir /path/to/golden_graphs_refined_1 \
  --memory-bank-dir /path/to/memory_bank \
  --output-dir /path/to/online_results
```

## CGME 生成的产物

当使用 `--output-dir /path/to/cgme_output` 时，离线 CGME 命令会将疾病图写入 `cgme_output/golden_graphs/`，将范例数据写入 `cgme_output/memory_bank/`。

可选的精化步骤可以生成精化图：

```bash
python scripts/refine_illness_graphs.py \
  --input-dir /path/to/cgme_output/golden_graphs \
  --output-dir /path/to/refined_graphs
```

图与记忆文件在离线输出与在线加载时使用相同的模式（schema）。路径设置请参阅 [配置](configuration.md)。
