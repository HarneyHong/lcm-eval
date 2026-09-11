# 数据收集与清洗指南（完整保留版）

> 阅读路线：第一部分完整保留原来的下载、普通 raw/JSON 收集、统一 parse、任务分配和注意事项。其余 18 库已有 raw 时无需重采；重新收集 IMDB/TPC-H-PK 时，不执行第一部分的两次独立 raw/JSON 命令，改执行第二部分的 paired 命令。第二部分同时说明如何生成单库对齐 master。

| 任务 | 应执行的章节 |
| --- | --- |
| 环境、下载、原任务分配 | 第一部分第 1～3、6 节 |
| 补采其余 18 库普通 raw | 第一部分第 4 节的 raw 命令 |
| 重采 IMDB/TPC-H-PK | 第二部分第 1～7 节 |
| 生成 Zero-Shot/DACE standard parsed | `TRAINING_GUIDE.md` 第二部分第 8 节 |

# 第一部分：原有完整收集流程（legacy/reference）

## 0. 产出（新目录）


| 阶段                    | 目录                                                                | 内容                           |
| --------------------- | ----------------------------------------------------------------- | ---------------------------- |
| raw                   | `data/raw_complex/<db>/complex_workload_200k_s1.json`             | EXPLAIN ANALYZE 文本（5000 条有效） |
| parsed                | `data/parsed_complex/<db>/complex_workload_200k_s1.json`          | 标准解析计划                       |
| baseline              | `data/parsed_complex_baseline/<db>/complex_workload_200k_s1.json` | MSCN/E2E/QueryFormer 用解析     |
| json（仅 imdb/tpc_h_pk） | `data/json_complex/<db>/complex_workload_200k_s1/...`             | QPP-Net 用                    |


## 1. 环境准备

```bash
git clone git@github.com:HarneyHong/lcm-eval.git
cd lcm-eval
python3 -m venv .venv && .venv/bin/pip install -r requirements/requirements.txt
# 其他：
# 安装 PostgreSQL
# .env 里 LOCAL_ROOT_PATH 指向仓库根目录
```

## 2. 下载数据

```bash
cd /data/workspace/lcm-eval/src
../.venv/bin/python download_from_osf.py --artifacts datasets   # -> data/datasets/<db>/*.csv
../.venv/bin/python download_from_osf.py --artifacts workloads  # -> data/workloads/training/<db>/complex_workload_200k_s1.sql
```

## 3. 收集数量约定


| 库             | cap_workload                             |
| ------------- | ---------------------------------------- |
| imdb、tpc_h_pk | **10000**（workload-driven 主目标，对齐论文已发布数据） |
| 其余所有库         | **5000**                                 |


## 4. 收集（每个库两步，每个人收集不同的库）

```bash
export LCM_ROOT=/data/workspace/lcm-eval
export PY="$LCM_ROOT/.venv/bin/python"
export DB=accidents      # 每次改为负责的数据库
export CSV_DB="$DB"      # tpc_h_pk 例外，应改成 tpc_h
export CAP=5000          # imdb/tpc_h_pk 为 10000，其余库为 5000

cd "$LCM_ROOT/src"
# 灌库
"$PY" run_benchmark.py --load_database \
  --data_dir "$LCM_ROOT/data/datasets/$CSV_DB" --dataset "$DB" --db_name "$DB" \
  --database_conn user=postgres,host=localhost

# 收集 raw
"$PY" run_benchmark.py --run_workload \
  --source "$LCM_ROOT/data/workloads/training/$DB/complex_workload_200k_s1.sql" --db_name "$DB" \
  --target "$LCM_ROOT/data/raw_complex/$DB/complex_workload_200k_s1.json" \
  --database_conn user=postgres,host=localhost --mode raw \
  --query_timeout 30 --repetitions_per_query 1 --cap_workload "$CAP"

# 旧独立 JSON 流程（仅保留作历史参考；新 IMDB/TPC-H-PK 不再执行）：
"$PY" run_benchmark.py --run_workload \
  --source "$LCM_ROOT/data/workloads/training/$DB/complex_workload_200k_s1.sql" --db_name "$DB" \
  --target "$LCM_ROOT/data/json_complex/$DB/complex_workload_200k_s1/complex_workload_200k_s1.json" \
  --database_conn user=postgres,host=localhost --mode json \
  --query_timeout 30 --repetitions_per_query 1 --cap_workload "$CAP"
```

## 5. 清洗（主机器统一执行）

```bash
cd "$LCM_ROOT/src"
"$PY" parse_all.py \
  --raw_dir "$LCM_ROOT/data/raw_complex" --parsed_plan_dir "$LCM_ROOT/data/parsed_complex" \
  --parsed_plan_dir_baseline "$LCM_ROOT/data/parsed_complex_baseline" \
  --workloads complex_workload_200k_s1 --cap_queries 10000
```

## 6. 任务分配


|     | 数据库                                                                |
| --- | ------------------------------------------------------------------ |
| 何沅东 | accidents, airline, baseball, basketball, carcinogenesis, consumer |
| 赵嘉祺 | credit, employee, fhnk, financial, geneea, genome                  |
| 房子珈 | hepatitis, imdb（10000）, movielens, seznam                          |
| 李昊森 | ssb, tournament, tpc_h_pk（10000）, walmart                          |


每人负责自己库的灌库 + raw 收集（imdb 归房子珈、baseball 归何沅东、tpc_h_pk 归李昊森，这三个需要额外 json 版）；完成后把 `data/raw_complex` 拷到主机器统一 parse。

## 7. 注意

- 200k 池按 cap 收集有效条数（>=100ms、非超时/失败），5000 条约数小时、10000 条约双倍；
- target 已存在会自动去重续跑，中断可重跑；
- `query_list` 会保留失败或过快的尝试，因此它的长度通常大于 cap；完成数量应以 parse 后的 `parsed_plans` 为准。


# 第二部分：当前 IMDB / TPC-H-PK paired 对齐收集流程

本指南生成 raw/JSON 成对候选以及 E2E/QueryFormer 的 10,000 条 baseline master。设计说明见 `ALIGNED_WORKLOAD_GUIDE.md`。

## 1. 环境变量

```bash
export LCM_ROOT=/data/workspace/lcm-eval
export PY="$LCM_ROOT/.venv/bin/python"
export PYTHONPATH="$LCM_ROOT/src"
```

每次选择一个数据库：

```bash
# IMDB
export DB=imdb
export CSV_DB=imdb

# TPC-H-PK 时改成：
# export DB=tpc_h_pk
# export CSV_DB=tpc_h
```

路径：

```bash
export WORKLOAD="$LCM_ROOT/data/workloads/training/$DB/complex_workload_200k_s1.sql"
export RAW="$LCM_ROOT/data/raw_complex/$DB/complex_workload_200k_s1_paired.json"
export JSON_RUN="$LCM_ROOT/data/json_complex/$DB/complex_workload_200k_s1_paired/complex_workload_200k_s1_paired.json"
export PARSED_STAGE="$LCM_ROOT/data/parsed_complex_baseline_staging/$DB/complex_workload_200k_s1.json"
export AUG_STAGE="$LCM_ROOT/data/augmented_complex_baseline_staging/$DB/complex_workload_200k_s1.json"
export BASELINE_MASTER="$LCM_ROOT/data/augmented_baseline_master/$DB/complex_workload_200k_s1.json"
export JSON_MASTER="$LCM_ROOT/data/json_master/$DB/complex_workload_200k_s1.json"
export ALIGN_DIR="$LCM_ROOT/data/alignment/$DB/complex_workload_200k_s1"
export COLUMN_STATS="$LCM_ROOT/src/cross_db_benchmark/datasets/$DB/column_statistics.json"
```

## 2. 灌库

数据库已经正确加载时跳过。TPC-H-PK 使用 `tpc_h_pk` schema，但 CSV 来自 `data/datasets/tpc_h`。

```bash
cd "$LCM_ROOT/src"
"$PY" run_benchmark.py --load_database \
  --data_dir "$LCM_ROOT/data/datasets/$CSV_DB" \
  --dataset "$DB" --db_name "$DB" \
  --database_conn user=postgres,host=localhost
```

## 3. 成对收集

这里特意使用带 `_paired` 后缀的新 target，避免覆盖旧实验数据。旧的独立 raw/JSON 文件没有 paired 元数据，不能直接续采；程序检测到它们也会拒绝继续写入。

```bash
export PAIR_CAP=12000

cd "$LCM_ROOT/src"
"$PY" run_benchmark.py --run_workload \
  --source "$WORKLOAD" --db_name "$DB" \
  --mode paired \
  --raw_target "$RAW" \
  --json_target "$JSON_RUN" \
  --database_conn user=postgres,host=localhost \
  --query_timeout 30 \
  --repetitions_per_query 1 \
  --min_query_ms 100 \
  --max_query_ms 30000 \
  --cap_workload "$PAIR_CAP"
```

`PAIR_CAP` 表示 pair-valid 候选数量，不表示最终 baseline master 数量。偶数 workload 行按 raw→JSON 执行，奇数行按 JSON→raw 执行。

## 4. raw → baseline staging

不要设置 `--cap_queries 10000`，否则会在 master 过滤之前提前截断。

```bash
mkdir -p "$(dirname "$PARSED_STAGE")"
cd "$LCM_ROOT/src"
"$PY" run_benchmark.py --parse_run \
  --source "$RAW" \
  --target "$PARSED_STAGE" \
  --parse_baseline \
  --min_query_ms 100 \
  --max_query_ms 30000
```

parser 会跳过 InitPlan/SubPlan 和其他无法解析的查询，并保留成功计划的 `query_id`。

## 5. 增加 sample bitmap

续采后必须使用 `force=True` 重建 staging 文件：

```bash
cd "$LCM_ROOT/src"
"$PY" - "$DB" "$LCM_ROOT/data/datasets/$CSV_DB" "$PARSED_STAGE" "$AUG_STAGE" <<'PY'
import sys
from models.workload_driven.preprocessing.sample_vectors import augment_sample_vectors

augment_sample_vectors(
    dataset=sys.argv[1],
    data_dir=sys.argv[2],
    plan_path=sys.argv[3],
    target_path=sys.argv[4],
    force=True,
)
PY
```

## 6. 生成 baseline/JSON master、manifest 和 split

```bash
cd "$LCM_ROOT/src"
"$PY" prepare_aligned_workload.py \
  --raw "$RAW" \
  --json "$JSON_RUN" \
  --augmented_baseline "$AUG_STAGE" \
  --baseline_master "$BASELINE_MASTER" \
  --json_master "$JSON_MASTER" \
  --alignment_manifest "$ALIGN_DIR/manifest.json" \
  --split_dir "$ALIGN_DIR/splits" \
  --column_statistics "$COLUMN_STATS" \
  --database "$DB" \
  --master_count 10000 \
  --seeds 0 1 2
```

若提示 master 不足，将 `PAIR_CAP` 增加 2,000，然后依次重跑第 3～6 节。paired collector 会从相同 target 按 `query_id` 断点续采。续采时 workload 内容、数据库名、timeout、runtime 边界、hint 和索引设置必须保持不变；程序会校验这些元数据并拒绝混采。

## 7. 完成检查

```bash
"$PY" - "$BASELINE_MASTER" "$JSON_MASTER" "$ALIGN_DIR/manifest.json" <<'PY'
import json, sys

baseline, json_path, manifest_path = sys.argv[1:]
with open(baseline) as f:
    baseline_ids = [p["query_id"] for p in json.load(f)["parsed_plans"]]
with open(json_path) as f:
    json_ids = [q["query_id"] for q in json.load(f)["query_list"]]
with open(manifest_path) as f:
    manifest = json.load(f)

assert manifest["ready"] is True
assert len(baseline_ids) == 10000
assert baseline_ids == json_ids == manifest["master_query_ids"]
print(json.dumps(manifest["summary"], indent=2))
PY
```

IMDB 和 TPC-H-PK 都通过后再开始训练。
