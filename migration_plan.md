# 算子归因迁移计划：Repo1 → Repo2

## 背景

- **Repo1** (`/data/hzy/Amadeus/amadeus/code`)：算子归因（QPOG）已优化
- **Repo2** (`/data/hzy/Amadeus/amadeus-collab/amadeus_core`)：检索（LLM walk决策 + evidence pool）已优化
- **目标**：把 Repo1 的算子归因系统原原本本迁移到 Repo2，保留 Repo2 的检索优化

---

## 文件级详细计划

### 1. `agents/base.py`（已完成）

**改了什么：**
| 内容 | 来源 | 说明 |
|------|------|------|
| `operator_guidelines` 类型 | Repo1 | `Dict[str, List[str]]` → `Dict[str, Dict[str, Dict]]`，每条规则带 score/count/confidence/reason |
| `_normalize_operator_guidelines()` | Repo1 新增 | 兼容旧 List 格式，统一转为带权重的 Dict 格式 |
| `_format_guidelines()` | Repo1 | 按 score 排序输出，支持 `max_guidelines_per_operator` 参数 |
| `update_guideline()` | Repo1 | 新增 `score`, `confidence`, `reason` 参数；重复规则累加 score 而非去重 |
| `get_operator_context()` | Repo1 新增 | optimizer backward pass 需要读取算子当前 guideline 上下文 |
| `find_text_anchors()` | Repo1 新增 | 在文本中定位短语的物理位置，QPOG 锚点系统依赖此方法 |
| `usage_stats` tracking | **Repo2 保留** | Repo1 没有这个，但 Repo2 的检索部分用它做 token 统计 |
| `max_guidelines_per_operator = 3` | Repo1 新增 | 限制每个算子最多保留的 guideline 数量 |

**为什么这么改：** QPOG 归因系统给每条 guideline 分配 credit（贡献度），需要带权重的结构来排序和淘汰低分规则，不能用简单 List。

---

### 2. `core/graph.py`

**要改的内容：**

| 内容 | 来源 | 说明 |
|------|------|------|
| `add_edge()` 返回值 | Repo1 | 原本无返回值 → 返回 `(source, target, key)` tuple。QPOG 构建 evidence 节点时需要精确 edge ref |
| `get_edge_ref(source, target, key)` | Repo1 新增 | 按精确 edge key 查询边属性 |
| `delete_edge_ref(source, target, key)` | Repo1 新增 | 按精确 edge key 删除单条边（区别于 `delete_edge` 删除两节点间所有边） |
| MemScenePool 集成 | **Repo2 保留** | `__init__` 中的 `self.scene_pool = MemScenePool(...)` 和 import |
| `get_full_state()` 签名 | **不改** | Repo1 加了 `max_chars=4000` 参数和截断逻辑，但这个跟归因无关，是 Repo1 独立的优化，**不迁移** |

**具体修改位置：**

```python
# 修改 add_edge 签名和返回值（第71行附近）
# 原：
def add_edge(self, source, target, relation, timestamp=None):
    ...
    self.graph.add_edge(source, target, relation=relation, timestamp=timestamp)
    # 无返回值

# 改为：
def add_edge(self, source, target, relation, timestamp=None) -> tuple:
    ...
    key = self.graph.add_edge(source, target, relation=relation, timestamp=timestamp)
    return (source, target, key)
    # 各 return 分支也要返回 (source, target, key)

# 在 delete_edge 后新增两个方法（第110行后）：
def get_edge_ref(self, source, target, key) -> Optional[dict]: ...
def delete_edge_ref(self, source, target, key) -> bool: ...
```

**不改的内容：**
- `get_full_state()` 保持 Repo2 当前版本（无 max_chars 参数）—— 截断是 Repo1 独立优化，与归因无关
- MemScenePool 相关的 `__init__` 和 import 全部保留
- 其他方法（`primitive_search`, `semantic_search`, `primitive_get_neighbors`, `primitive_read`, `save`, `load`）不动

---

### 3. `agents/answerer.py`

**要改的内容：**

| 内容 | 来源 | 说明 |
|------|------|------|
| `TRACE_PAYLOAD_KEYS` 常量 | Repo1 新增 | trace 输出的字段名列表 |
| `__init__` 新增参数 | Repo1 | 加 `trace_config`, 初始化 `self.last_trace`, `self._answer_lock` |
| trace 基础方法 | Repo1 新增 | `_normalize_trace_config`, `_new_trace_payload`, `_record_usage`, `_safe_preview`, `_build_merged_answer_input`, `_set_retrieved_items_trace`, `_finalize_trace`, `_write_trace_file`, `_stable_question_slug` |
| `_append_operator_trace()` | Repo1 新增 | 核心方法：记录每步算子执行的输入/输出/锚点/inner_monologue |
| `_build_node_anchor_payload()` | Repo1 新增 | 构建节点锚点载荷 |
| `answer()` 支持 `return_trace` | Repo1 | 加 `return_trace=True` 时返回 `(answer, trace_dict)` |
| 检索逻辑 | **Repo2 保留** | LLM决策walk、evidence pool、MemScene两级检索、`_answer_internal` 全部保留 |
| `_parse_json_response` | **Repo2 保留** | Repo1 是内联解析，Repo2 抽了公共方法 |
| `_normalize_fact`, `_merge_evidence_pool`, `_render_evidence_pool` | **Repo2 保留** | evidence pool 相关方法 |
| `_extract_step_evidence` | **Repo2 保留** | 每步提取相关事实 |
| `_render_candidate_options`, `_validate_walk_steps`, `_fallback_walk_selection` | **Repo2 保留** | LLM walk 决策相关 |

**具体修改位置：**

```python
# 1. 文件头新增 import（保留 Repo2 已有的，新增 threading, os, pathlib）
import threading
import os
from pathlib import Path

# 2. 类定义前新增常量
TRACE_PAYLOAD_KEYS = (
    "retrieved_items_with_ranks", "supporting_fact_scene_membership",
    "merged_answer_input", "api_call_count", "prompt_tokens",
    "completion_tokens", "total_tokens", "operator_trace",
)

# 3. __init__ 新增（在现有参数后追加）
def __init__(self, graph, model_name, api_base, api_key, trace_config=None):
    ...  # 保留现有初始化
    self.trace_config = self._normalize_trace_config(trace_config)
    self.last_trace = self._new_trace_payload()
    self._answer_lock = threading.Lock()

# 4. 新增所有 trace 方法（作为类方法插入，不影响现有方法）

# 5. 在 _answer_internal 的关键位置插入 _append_operator_trace 调用：
#    - 初始 SEARCH 完成后
#    - 每次 WALK 步骤后
#    - READ 返回答案时
#    - fallback 返回时

# 6. answer() 方法加 return_trace 参数
def answer(self, question, return_trace=False):
    ...
    result = self._answer_internal(question)
    if return_trace:
        return result, dict(self.last_trace)
    return result
```

**不改的内容：**
- `_answer_internal` 的检索和 walk 决策逻辑全部保留
- evidence pool 系列方法全部保留
- MemScene 两级检索全部保留
- `_llm_extract_keywords`, `_keyword_search`, 各种 `_score_*`, `_edge_*_search` 全部保留
- `scene_top_n`, `walk_candidate_top_k` 等配置参数全部保留

---

### 4. `engine/optimizer.py`

**要改的内容：**

| 内容 | 来源 | 说明 |
|------|------|------|
| 整个文件 | **Repo1 替换** | 用 Repo1 的 1359 行版本整体替换 Repo2 的 462 行版本 |

**替换后需要适配的内容：**

| 适配项 | 说明 |
|--------|------|
| import 路径 | `amadeus.code.agents.*` → `amadeus_core.agents.*` |
| MemScenePool import | 加 `from amadeus_core.core.scene import MemScenePool` |
| `step()` 签名 | 保留 Repo2 的 `affected_scene_ids` + `scene_pool` 参数，在 Repo1 版本基础上加入 |
| `_process_single_duel()` | 调用 `self.answerer.answer(question, return_trace=True)` 获取 trace |
| `_evaluate_and_update` 中 scene 相关 | Repo1 版本已经不需要 scene 参数（归因走 QPOG），但 `force_update` 调用需要适配 scene |

**Repo1 optimizer.py 包含的核心归因模块（全部迁移）：**

1. **QPOG 构建** (`_build_question_operator_graph`)：从 answerer trace 构建 Question-Operator Graph
2. **文本梯度传播** (`_propagate_text_gradient`)：Judge failure reason 沿图反向传播，每层经 inner_monologue 翻译
3. **修复子图选择** (`_select_repair_subgraph`)：LLM 逐节点投票决定哪些算子需要修复
4. **物理回溯** (`_physical_backtrace`)：基于锚点评分找到决定性步骤
5. **反向传播** (`_backward_pass`)：ADOPT 风格的链式梯度分解
6. **QPOG 保存** (`_save_qpog_graph`)：序列化保存用于分析
7. **更新计划** (`_build_update_plan`, `_apply_update_plan`)：多算子更新的归一化和应用

---

## 不涉及的文件

- `agents/questioner.py` — 不改
- `agents/builder.py` — 不改（`force_update` 接口两边一致）
- `core/schema.py` — 不改
- `core/buffer.py` — 不改
- `core/scene.py` — 不改
- `prompts/` 目录 — 不改
- `engine/__init__.py` — 不改

---

## 风险点

1. **builder.force_update 接口兼容**：Repo1 的 `force_update(graph_patch)` 传的是 list，Repo2 的 `force_update(str, scene_id, scene_pool)` 传的是字符串。需要确认 builder 两边接口是否一致。
2. **answerer trace 插入位置**：Repo2 的 `_answer_internal` 逻辑和 Repo1 差异较大（Repo2 有 LLM walk 决策、evidence pool），trace 插入点需要仔细对齐。
3. **optimizer 的 `_process_single_duel` 中 answerer 调用方式**：Repo1 用 `self.answerer.answer(question, return_trace=True)`，Repo2 用 `self.answerer._answer_internal(question)`。需要统一为支持 trace 的调用方式。
