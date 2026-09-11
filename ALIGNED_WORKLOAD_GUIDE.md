# Raw/JSON 对齐采集与公平训练说明

本说明用于在 IMDB 和 TPC-H-PK 上为 E2E、QueryFormer 和 QPP-Net 重新准备数据。目标是固定一份 E2E/QueryFormer 可用的 10,000 条 baseline master，再在相同 split 内标记 QPP-Net 的支持范围。

## 1. 实验口径

每条候选 SQL 分别执行一次 raw EXPLAIN ANALYZE 和 JSON EXPLAIN ANALYZE，偶数行按 raw→JSON、奇数行按 JSON→raw 执行。两边不共享 runtime 标签：

- E2E、QueryFormer 使用 raw runtime；
- QPP-Net 使用 JSON runtime；
- 只要求 SQL、物理计划和 split 一致。

一条查询只有满足以下条件才是 `pair_valid`：

```text
raw/JSON 都执行成功且没有超时
raw/JSON runtime 都在 [100ms, 30000ms]
raw/JSON 都不是零基数
raw/JSON 规范化后的物理计划完全一致
```

之后在 `pair_valid` 基础上继续要求，形成baseline master样本：

```text
不含 InitPlan/SubPlan
raw → baseline parsed 成功
sample bitmap 已生成
满足 E2E/QueryFormer 的结构输入限制
```

QPP-Net 的算子或编码限制不决定 baseline master。master 固定并划分后，才在每个 split 内应用 QPP-Net support mask。

现有数据中已观察到的主要 QPP-Net 不支持项包括 `Memoize`、`Limit`、`BitmapOr`，以及带子节点、无法满足当前 QPP-Net 输入维度假设的 `Result`。代码不会改写或删除这些 SQL；它会在 master 固定后逐条检查实际 JSON 输入，把具体原因写入 manifest。

## 2. 代码改动

- `run_benchmark.py`：增加 `--mode paired（两者一起收集）`、`--raw_target（raw格式的输出目录）`、`--json_target（json格式的输出目录）`。
- `paired_workload.py`：在同一 SQL 循环中收集 raw/JSON，生成稳定 `query_id`，执行双边 runtime/基数检查并支持断点续采。
- `plan_fingerprint.py`：统一文本和 JSON 的算子表示并比较物理计划。
- `parse_plan.py`：把 `query_id` 和 `workload_index` 传递到 parsed plan。
- `prepare_aligned_workload.py`：生成 augmented baseline master、对应 JSON master、alignment manifest 和三个 seed 的 split。
- `aligned_splits.py`：按 query ID 加载固定 split，并在 `qpp_native`/`matched` 协议中应用 QPP-Net mask。
- baseline/QPP-Net dataloader：启用 manifest split；旧命令未传 manifest 时保持原行为。
- QPP-Net collator：编码失败立即报出 query ID，不再静默丢样本。
- `aligned_metrics.py`：报告 coverage，并验证 strict matched 的预测 query ID 集合。

## 3. 数据目录

不创建重复的 `paired_complex`。成对采集仍写原始目录：

```text
data/raw_complex/<db>/complex_workload_200k_s1_paired.json
data/json_complex/<db>/complex_workload_200k_s1_paired/complex_workload_200k_s1_paired.json
```

派生产物：

```text
data/parsed_complex_baseline_staging/<db>/complex_workload_200k_s1.json
data/augmented_complex_baseline_staging/<db>/complex_workload_200k_s1.json
data/augmented_baseline_master/<db>/complex_workload_200k_s1.json
data/json_master/<db>/complex_workload_200k_s1.json
data/alignment/<db>/complex_workload_200k_s1/manifest.json
data/alignment/<db>/complex_workload_200k_s1/splits/seed_{0,1,2}.json
```

`manifest.json` 记录 pair 状态、baseline 排除原因、QPP-Net 支持状态和 coverage。原始计划只保存在 raw/json 文件中。

## 4. 收集与生成 master

完整命令见 `COLLECTION_GUIDE.md`。推荐先以 `--cap_workload 12000` 收集 pair-valid 候选。随后运行 raw parse、sample bitmap 和 master preparation。

若 preparation 报告 baseline master 不足 10,000 条：

1. 把 paired cap 增加 2,000；
2. 用相同 raw/JSON target 重跑 paired 命令，程序按 query ID 续采；
3. 重新生成 staging parsed 和 sample bitmap；
4. 再运行 preparation；
5. 直到 manifest 中 `ready=true`、`baseline_master=10000`。

## 5. Split 和训练协议

每个 seed 使用 `SHA256(seed:query_id)` 稳定排序，先在 baseline master 上固定 8,000/1,000/1,000：

```text
baseline_native:
  E2E/QueryFormer 使用完整 master train/validation/test

qpp_native:
  QPP-Net 在原 split 内仅使用 qppnet_supported=true 的 ID
  同时报告 train/validation/test coverage

matched:
  E2E、QueryFormer、QPP-Net 全部使用原 split 与 QPP support mask 的交集
  三个模型必须重新训练，不能只过滤测试预测
```

完整训练命令见 `TRAINING_GUIDE.md`。

## 6. 完成标准

```text
baseline master 数量 = 10000
baseline master query ID 顺序 = JSON master query ID 顺序
master split = 8000/1000/1000，且三个集合互斥
matched train/validation/test 均为对应 master split 的子集
strict matched 三个模型预测 CSV 的 query ID 集合完全一致
E2E/QueryFormer 标签来自 raw，QPP-Net 标签来自 JSON
```
