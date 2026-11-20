# ECR-Agent 架构

[ECR-Agent](https://arxiv.org/abs/2601.06636) 将**动态因果推理（Dynamic Causal Inference，DCI）**与**批评者驱动的图与记忆演化（Critic-Driven Graph & Memory Evolution，CGME）**相结合。CGME 从支持病例中构建可复用的医学知识与经验；DCI 则利用这些资源对新临床病例进行推理。

## 组件

| 组件 | 主要实现 | 职责 |
| --- | --- | --- |
| 双通路感知（Dual-Pathway Perception） | `Phase1Manager`、`phase1_perception.py`、`phase1_trackB.py` | 生成鉴别诊断候选以及结构化的病例表示。 |
| 因果图初始化（Causal Graph Initialization） | `MedicalGraph`、`Phase2HybridEngine`、`GoldenGraphLoader` | 利用观察、候选诊断以及可用的疾病图来初始化患者特异的因果图。 |
| 前向因果推理（Forward Causal Reasoning） | `Phase2GraphReasoning`、`Phase2HybridEngine`、`PubMedTool`、`OpenTargetsTool` | 检索医学证据并将其与候选疾病建立关联。 |
| 反向因果推理（Backward Causal Reasoning） | Phase 2 推理模块、`MedicalGraph.create_shadow_p_node` | 检查期望证据，并用 Shadow 节点表示未能匹配的观察。 |
| 证据审计（Evidence Audit） | `graph_tools.py`、`JudgeAgent`、`MemoryAugmentedJudge` | 对图证据进行评分与总结；在线推理还会检索相似的范例。 |
| 批评者驱动的演化（Critic-Driven Evolution） | `OfflineTrainer`、`ConcurrentOfflineTrainer`、`CriticalModelAgent` | 通过批评者反馈评估并精化对支持病例的推理。 |
| 疾病图演化（Illness Graph Evolution） | `AggregationManager`、`CanonicalGraph`、`scripts/refine_illness_graphs.py` | 按疾病聚合已验证的证据，并支持可选的图精化。 |
| 范例库（Exemplar Base） | `MemoryBankManager`、`EmbeddingClient` | 维护病例记录与嵌入向量，用于经验检索。 |

## 离线 CGME

```text
参考集 + 853 例支持集清单（support manifest）
  → scripts/run_cgme.py
  → 双通路感知
  → Phase2GraphReasoning（实时证据）
  → JudgeAgent 与批评者引导的精化
  → 疾病图聚合 + 范例库
  → 可选 scripts/refine_illness_graphs.py
```

离线 CGME 从支持病例中构建疾病图，而不是依赖预构建的疾病图。其成功病例会被纳入图集合与范例库。

## 在线 DCI

```text
基准病例 + 预构建疾病图 + 范例库
  → scripts/run_ecr_agent.py
  → 双通路感知
  → Phase2HybridEngine（图初始化 + 实时证据）
  → 前向与反向因果推理
  → 图评分与证据审计
  → 最终诊断与病例级输出
```

在线 DCI 使用生成的疾病图来初始化推理，同时持续检索相关证据。图加载器优先使用 `*_Refined_GG.json` 文件，在可用时也可使用原始的 `*_GG.json` 文件。如果某个候选疾病没有预构建的疾病图，混合推理路径仍可依靠实时证据检索继续运行。

患者观察被表示为 **Present（存在）**、**Absent（不存在）** 或 **Missing（缺失）**。知识节点包括 **Pivot（枢纽）** 和 **General（一般）** 证据，而 **Shadow（影子）** 节点则捕获被期望但未能匹配的观察。范例库包含紧凑的病例记录以及用于检索的对齐嵌入向量。

关于相关的输入格式、模型设置与产物布局，请参阅 [数据集接口](dataset_interface.md)、[配置](configuration.md) 与 [产物](artifacts.md)。
