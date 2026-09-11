# 当前数据收集后的完整训练指南

本指南只描述当前要执行的流程，不再保留旧的 raw/JSON 独立采集训练命令。当前状态是：

- 20 个数据库的旧 raw 已经收集完成；
- 其余 18 库继续使用现有 raw；
- JOB 在仓库中的数据库名是 `imdb`；
- `imdb` 和 `tpc_h_pk` 旧 raw/JSON 的 SQL 集合没有对齐，因此这两个库需要按 paired 模式重新收集；
- E2E、QueryFormer、QPP-Net 在 `imdb` 和 `tpc_h_pk` 上做单库训练；
- Zero-Shot、DACE 使用 20 库 standard parsed，按 19 库训练、1 库测试。

| 负责人 | 模型 | 当前训练输入 | 协议 |
| --- | --- | --- | --- |
| 房子珈 | E2E | paired raw 派生的 augmented baseline master | 单库固定 80/10/10 |
| 李昊森 | QueryFormer | 与 E2E 相同的 baseline master | 单库固定 80/10/10 |
| 何沅东 | QPP-Net | paired JSON 派生的 JSON master | 单库固定 split 内的 QPP-supported 子集 |
| 赵嘉祺 | Zero-Shot / DACE | 20 库 raw 派生的 standard parsed | 其余 19 库训练、目标库测试 |

## 1. 其余 18 库是否需要重新收集

不需要因为 IMDB/TPC-H-PK 的 raw/JSON 错位而重新收集其余 18 库。那 18 个库不参加本次 QPP-Net 的 paired JSON 对齐，它们已有的 raw 格式可以继续用于 Zero-Shot/DACE。

但是需要区分两层过滤：

```text
raw collector 的有效性：执行成功、未超时、raw runtime >= 100ms、非零基数
standard parser 的有效性：在上述基础上还可能跳过 InitPlan/SubPlan、空 Result 或解析失败
```

所以，旧 raw 收集到 5,000 条有效查询，不保证 standard parse 后仍有 5,000 条。正确处理方式是：

1. 不整库重采；
2. 先按第 7.2 节对 20 库生成 standard parsed 并计数；
3. 只有某库少于目标数量时，才按第 7.3 节在原 raw target 上提高 cap、断点增量补采；
4. 补采后重新 parse，直到其余库各 5,000 条，IMDB/TPC-H-PK 各 10,000 条。

## 2. 新旧数据格式与五个模型的关系

新的 paired raw 没有改变 PostgreSQL raw 计划的核心格式，只在原来的每条 `query_list` 记录上增加了：

```text
query_id
workload_index
execution_order
pair_valid / pair_invalid_reasons
raw_runtime_ms / json_runtime_ms
raw/json plan fingerprint
```

因此 IMDB/TPC-H-PK 的新 paired raw 可以和其余 18 库旧 raw 一样交给 standard parser。Zero-Shot/DACE 不使用的附加字段会被忽略。

| 模型 | 原始来源 | 真正训练输入 | 标签 |
| --- | --- | --- | --- |
| E2E | IMDB/TPC-H-PK paired raw | `augmented_baseline_master` | raw runtime |
| QueryFormer | IMDB/TPC-H-PK paired raw | 与 E2E 相同 | raw runtime |
| QPP-Net | IMDB/TPC-H-PK paired JSON | `json_master` 中 QPP-supported 查询 | JSON runtime |
| Zero-Shot | 20 库 raw | `parsed_complex` standard parsed | raw runtime |
| DACE | 20 库 raw | 与 Zero-Shot 相同 | raw runtime |

不要把下面几种文件混用：

```text
baseline master：只给 E2E / QueryFormer
JSON master：只给 QPP-Net
standard parsed：给 Zero-Shot / DACE
alignment manifest / split：当前只控制三个单库模型
```

## 3. 一次性 CPU 环境

### 3.1 已有 Python 环境时

如果训练机已经有可用的 `.venv`，先执行：

```bash
cd /data/workspace/lcm-eval/src
/data/workspace/lcm-eval/.venv/bin/python -c 'import torch, dgl, gensim; print("environment OK")'
```

作用：检查 PyTorch、DGL 和 word2vec 依赖能否导入。如果打印 `environment OK`，可以跳过下一节。项目的 `main.py` 在第 3.3 节设置好 `NODE00...NODE05` 后再检查。

### 3.2 没有环境时

 参考 [https://github.com/fzj2007/LCM-Reproduction-Script](https://github.com/fzj2007/LCM-Reproduction-Script) 已验证过 CPU 版 PyTorch 2.3.0 + DGL 1.1.3。在 `/data/workspace/lcm-eval` 中建立独立环境：

```bash
cd /data/workspace/lcm-eval
python3 -m venv .venv-cpu

export PY=/data/workspace/lcm-eval/.venv-cpu/bin/python

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install dgl==1.1.3
"$PY" -m pip install \
  attrs==25.3.0 filelock==3.16.1 gensim==4.3.3 joblib==1.4.2 lightgbm==4.6.0 \
  loralib==0.1.2 networkx==3.1 numpy==1.24.4 optuna==4.5.0 \
  pandas==2.0.3 psutil==7.0.0 psycopg2-binary==2.9.10 \
  pydantic==2.10.6 python-dotenv==1.0.1 PyYAML==6.0.3 \
  scikit-learn==1.3.2 scipy==1.10.1 seaborn==0.13.2 \
  torchdata==0.9.0 tqdm==4.67.1 wandb==0.21.1 osfclient==0.0.5
```

作用：建立 CPU 训练环境，不会执行训练。建议使用 Python 3.9～3.11；如果系统默认是 Python 3.12，请先安装 Python 3.10 或 3.11。

### 3.3 每次登录后设置变量

```bash
export LCM_ROOT=/data/workspace/lcm-eval
export SRC="$LCM_ROOT/src"

# 仓库中已经有 .venv 时使用这一行：
export PY="$LCM_ROOT/.venv/bin/python"

# 只有按第 3.2 节新建了 .venv-cpu 时，才注释上一行并改用这一行：
# export PY="$LCM_ROOT/.venv-cpu/bin/python"

export RAW_ROOT="$LCM_ROOT/data/raw_complex"
export JSON_ROOT="$LCM_ROOT/data/json_complex"
export STANDARD_ROOT="$LCM_ROOT/data/parsed_complex"
export SENT_ROOT="$LCM_ROOT/data/sentences"
export STATS_ROOT="$LCM_ROOT/data/feature_statistics"
export MODEL_ROOT="$LCM_ROOT/data/models"
export EVAL_ROOT="$LCM_ROOT/data/evaluation"

export PYTHONHASHSEED=0
export DGLBACKEND=pytorch
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# 模型配置代码会读取 6 个集群节点变量。本机不会连接这些节点，
# 但导入代码时变量必须是合法 JSON，所以统一填本机占位值。
NODE_ENV='{"hostname":"localhost","python":"3.10"}'
export NODE00="$NODE_ENV" NODE01="$NODE_ENV" NODE02="$NODE_ENV"
export NODE03="$NODE_ENV" NODE04="$NODE_ENV" NODE05="$NODE_ENV"

mkdir -p "$STANDARD_ROOT" "$SENT_ROOT" "$STATS_ROOT" "$MODEL_ROOT" "$EVAL_ROOT"

cd "$SRC"
"$PY" -c 'import main; print("main.py OK")'
```

最后打印 `main.py OK` 即表示环境可以用。


## 4. 当前输入检查

先执行第 3.3 节的公共环境变量，再定义 20 库及其 raw 路径规则：

```bash
ALL_DBS=(
  accidents airline baseball basketball carcinogenesis consumer
  credit employee fhnk financial geneea genome hepatitis imdb
  movielens seznam ssb tournament tpc_h_pk walmart
)

raw_source() {
  local db="$1"
  if [[ "$db" == imdb || "$db" == tpc_h_pk ]]; then
    printf '%s\n' "$RAW_ROOT/$db/complex_workload_200k_s1_paired.json"
  else
    printf '%s\n' "$RAW_ROOT/$db/complex_workload_200k_s1.json"
  fi
}

missing=0
for db in "${ALL_DBS[@]}"; do
  source=$(raw_source "$db")
  if [[ -s "$source" ]]; then
    echo "OK      $db  $source"
  else
    echo "MISSING $db  $source"
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || { echo 'raw 文件未收齐，停止'; false; }
```

完成 `COLLECTION_GUIDE.md` 中的重新收集和 master preparation 后，IMDB/TPC-H-PK 还必须检查 paired JSON 和对齐产物：

```bash
missing=0
for db in imdb tpc_h_pk; do
  json="$JSON_ROOT/$db/complex_workload_200k_s1_paired/complex_workload_200k_s1_paired.json"
  baseline="$LCM_ROOT/data/augmented_baseline_master/$db/complex_workload_200k_s1.json"
  json_master="$LCM_ROOT/data/json_master/$db/complex_workload_200k_s1.json"
  alignment="$LCM_ROOT/data/alignment/$db/complex_workload_200k_s1/manifest.json"
  for file in "$json" "$baseline" "$json_master" "$alignment"; do
    [[ -s "$file" ]] && echo "OK      $file" || { echo "MISSING $file"; missing=1; }
  done
done
[[ "$missing" -eq 0 ]] || { echo 'paired 派生产物不完整，停止'; false; }
```

训练机只需要上述文件、两个目标库的 CSV 以及代码环境，不需要连接 PostgreSQL。只有收集机增量补采时才需要 PostgreSQL。

## 5. 准备 IMDB/TPC-H-PK 对齐 master

先按 `COLLECTION_GUIDE.md` 第二部分对 IMDB 和 TPC-H-PK 完成 paired 收集、baseline parse、sample bitmap 和 master preparation。最终必须满足：

```text
baseline master = 10,000 条
baseline master query_id 顺序 = JSON master query_id 顺序
seed 0/1/2 的 master split = 8,000 / 1,000 / 1,000
QPP-Net 支持状态只作为 split 内 mask，不决定 baseline master
```

选择一个目标库和 seed：

```bash
export DB=imdb       # 第二个目标库改为 tpc_h_pk
export SEED=0        # 完成后依次改为 1、2

export BASELINE_MASTER="$LCM_ROOT/data/augmented_baseline_master/$DB/complex_workload_200k_s1.json"
export JSON_MASTER="$LCM_ROOT/data/json_master/$DB/complex_workload_200k_s1.json"
export ALIGN_DIR="$LCM_ROOT/data/alignment/$DB/complex_workload_200k_s1"
export ALIGNMENT="$ALIGN_DIR/manifest.json"
export SPLIT="$ALIGN_DIR/splits/seed_$SEED.json"
export COLUMN_STATS="$SRC/cross_db_benchmark/datasets/$DB/column_statistics.json"
export SENTENCES="$LCM_ROOT/data/sentences/aligned/$DB/sentences.json"
export WORD2VEC="$LCM_ROOT/data/sentences/aligned/$DB/word2vec.m"
export BASELINE_STATS="$LCM_ROOT/data/feature_statistics/aligned_baseline/$DB/feature_statistics.json"
export QPP_STATS="$LCM_ROOT/data/feature_statistics/aligned_qpp/$DB/feature_statistics.json"
```

检查 query ID 和数量：

```bash
"$PY" - "$BASELINE_MASTER" "$JSON_MASTER" "$ALIGNMENT" "$SPLIT" <<'PY'
import json, sys
baseline_path, json_path, alignment_path, split_path = sys.argv[1:]
with open(baseline_path) as f:
    baseline_ids = [p["query_id"] for p in json.load(f)["parsed_plans"]]
with open(json_path) as f:
    json_ids = [q["query_id"] for q in json.load(f)["query_list"]]
with open(alignment_path) as f:
    alignment = json.load(f)
with open(split_path) as f:
    split = json.load(f)
split_ids = split["train"] + split["validation"] + split["test"]
assert alignment["ready"] is True
assert len(baseline_ids) == 10000
assert baseline_ids == json_ids == alignment["master_query_ids"]
assert len(split["train"]) == 8000
assert len(split["validation"]) == 1000
assert len(split["test"]) == 1000
assert set(split_ids) == set(baseline_ids) and len(split_ids) == len(set(split_ids))
print(json.dumps(alignment["summary"], indent=2))
PY
```

## 6. E2E、QueryFormer、QPP-Net 单库训练

这里有两种结果口径：

```text
native：E2E/QueryFormer 使用完整 master split；QPP-Net 使用各 split 内自身支持的查询
matched：三个模型都使用各 master split 与 QPP support mask 的交集，并分别重新训练
```

不存在公共 runtime 标签。E2E/QueryFormer 使用 raw runtime，QPP-Net 使用 JSON runtime；公平性来自相同 SQL、相同物理计划和相同 query ID split。


### 6.1 生成公共预处理资源

#### 6.1.1 sentences 和 word2vec

TPC-H-PK 的 CSV 目录是 `tpc_h`：

```bash
export CSV_DB="$DB"
[[ "$DB" == "tpc_h_pk" ]] && export CSV_DB=tpc_h

mkdir -p "$(dirname "$SENTENCES")"
cd "$SRC"
"$PY" - "$DB" "$BASELINE_MASTER" "$LCM_ROOT/data/datasets/$CSV_DB" "$SENTENCES" <<'PY'
import sys
from models.workload_driven.preprocessing.sentence_creation import create_sentences

create_sentences(
    dataset=sys.argv[1],
    plan_paths=[sys.argv[2]],
    data_dir=sys.argv[3],
    target=sys.argv[4],
)
PY

"$PY" - "$SENTENCES" "$WORD2VEC" <<'PY'
import sys
from models.workload_driven.preprocessing.word_embeddings import compute_word_embeddings
compute_word_embeddings(sys.argv[1], sys.argv[2])
PY
```

#### 6.1.2 feature statistics

```bash
cd "$SRC"
"$PY" - "$BASELINE_MASTER" "$BASELINE_STATS" "$JSON_MASTER" "$QPP_STATS" <<'PY'
import sys
from training.preprocessing.feature_statistics import gather_feature_statistics

baseline, baseline_stats, json_master, qpp_stats = sys.argv[1:]
gather_feature_statistics([baseline], baseline_stats)
gather_feature_statistics([json_master], qpp_stats)
PY
```

### 6.2 查看 QPP-Net coverage

```bash
cd "$SRC"
"$PY" evaluation/aligned_metrics.py \
  --alignment_manifest "$ALIGNMENT" \
  --split_manifest "$SPLIT"
```

它会分别输出 train、validation、test 中 QPP-Net 支持的数量和比例。

### 6.3 Native 训练

#### 6.3.1 E2E

```bash
export MODEL_DIR="$LCM_ROOT/data/models/baseline_native/e2e/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/baseline_native/e2e/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py --mode train --model_type e2e --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol baseline_native \
  --num_workers 0 --seed "$SEED" \
  2>&1 | tee "$EVAL_DIR/train_seed_${SEED}.log"
```

#### 6.3.2 QueryFormer

```bash
export MODEL_DIR="$LCM_ROOT/data/models/baseline_native/query_former/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/baseline_native/query_former/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py --mode train --model_type query_former --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol baseline_native \
  --num_workers 0 --seed "$SEED" \
  2>&1 | tee "$EVAL_DIR/train_seed_${SEED}.log"
```

#### 6.3.3 QPP-Net

```bash
export MODEL_DIR="$LCM_ROOT/data/models/qpp_native/qppnet/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/qpp_native/qppnet/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py --mode train --model_type qppnet --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$QPP_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --workload_runs "$JSON_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol qpp_native \
  --num_workers 0 --seed "$SEED" \
  2>&1 | tee "$EVAL_DIR/train_seed_${SEED}.log"
```

Native 结果反映各自原生可用范围；QPP-Net 结果必须连同 coverage 报告，不能直接作为严格 matched 对比。

### 6.4 Strict matched 训练

严格比较时，三个模型都使用 `--experiment_protocol matched`，并使用独立模型/结果目录重新训练：

```text
data/models/matched/e2e/<db>
data/models/matched/query_former/<db>
data/models/matched/qppnet/<db>
data/evaluation/matched/e2e/<db>
data/evaluation/matched/query_former/<db>
data/evaluation/matched/qppnet/<db>
```

输入仍然是：

```text
E2E/QueryFormer：BASELINE_MASTER
QPP-Net：JSON_MASTER
```

三个 dataloader 都会在原 master split 内应用同一 QPP support mask。不能复用第 6.3 节使用完整 baseline train 训练出的 E2E/QueryFormer checkpoint。

#### 6.4.1 Matched E2E

```bash
export MODEL_DIR="$LCM_ROOT/data/models/matched/e2e/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/matched/e2e/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py --mode train --model_type e2e --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol matched \
  --num_workers 0 --seed "$SEED" \
  2>&1 | tee "$EVAL_DIR/train_seed_${SEED}.log"
```

#### 6.4.2 Matched QueryFormer

```bash
export MODEL_DIR="$LCM_ROOT/data/models/matched/query_former/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/matched/query_former/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py --mode train --model_type query_former --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol matched \
  --num_workers 0 --seed "$SEED" \
  2>&1 | tee "$EVAL_DIR/train_seed_${SEED}.log"
```

#### 6.4.3 Matched QPP-Net

```bash
export MODEL_DIR="$LCM_ROOT/data/models/matched/qppnet/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/matched/qppnet/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py --mode train --model_type qppnet --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$QPP_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --workload_runs "$JSON_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol matched \
  --num_workers 0 --seed "$SEED" \
  2>&1 | tee "$EVAL_DIR/train_seed_${SEED}.log"
```

### 6.5 验证 strict matched 结果

三个模型都完成后：

```bash
cd "$SRC"
"$PY" evaluation/aligned_metrics.py \
  --alignment_manifest "$ALIGNMENT" \
  --split_manifest "$SPLIT" \
  --prediction "e2e=$LCM_ROOT/data/evaluation/matched/e2e/$DB/complex_workload_200k_s1_${SEED}_test_pred.csv" \
  --prediction "query_former=$LCM_ROOT/data/evaluation/matched/query_former/$DB/complex_workload_200k_s1_${SEED}_test_pred.csv" \
  --prediction "qppnet=$LCM_ROOT/data/evaluation/matched/qppnet/$DB/complex_workload_200k_s1_${SEED}_test_pred.csv"
```

工具会拒绝 query ID 缺失、额外或不一致的结果，并分别基于模型自己的标签计算 Q-error：

- E2E/QueryFormer label 是 raw runtime；
- QPP-Net label 是 JSON runtime；
- 不要求两种 label 数值相同。


## 7. Zero-Shot、DACE 跨库训练

Zero-Shot/DACE 不直接读取 raw，而是读取 standard parsed。对于 IMDB/TPC-H-PK，新 paired raw 与旧 raw 的核心格式相同，所以可以和其余 18 库一起转换和跨库训练。

这两个模型沿用原来的 workload-agnostic 协议：每次选择一个目标库，其余 19 库训练，目标库整个 standard workload 测试。不要给下面的命令传 `--split_manifest`、`--alignment_manifest` 或 `--experiment_protocol`。

由于 IMDB 和 TPC-H-PK 同时属于这 20 库，无论哪个是测试目标，训练集合或测试集合都发生了变化，所以 Zero-Shot 和 DACE 的两个目标库、三个 seed 都必须重新训练。下面使用新的 `cross_db_paired` 模型目录，避免程序误加载旧 checkpoint。

### 7.1 检查 20 库 raw

执行第 4 节中的 `ALL_DBS`、`raw_source` 和检查命令。全部为 `OK` 后继续。

### 7.2 生成并审核 20 库 standard parsed

```bash
for db in "${ALL_DBS[@]}"; do
  expected=5000
  [[ "$db" == imdb || "$db" == tpc_h_pk ]] && expected=10000

  source=$(raw_source "$db")
  target="$STANDARD_ROOT/$db/complex_workload_200k_s1.json"
  mkdir -p "$(dirname "$target")"

  cd "$SRC"
  "$PY" run_benchmark.py --parse_run \
    --source "$source" \
    --target "$target" \
    --parse_join_conds \
    --min_query_ms 100 \
    --max_query_ms 30000 \
    --cap_queries "$expected"

  count=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["parsed_plans"]))' "$target")
  echo "$db: $count/$expected"
done
```

预期数量：

```text
其余 18 库：各 5,000 条
IMDB：10,000 条
TPC-H-PK：10,000 条
```

对 IMDB/TPC-H-PK 做 standard parse 时不检查 `pair_valid`，只应用和其余 18 库相同的 raw-side 规则。否则会形成“18 库 raw-only、2 库 raw+JSON”的不一致跨库筛选。

### 7.3 某个旧库 parse 后数量不足时

不删除旧文件，也不从头重采。在原收集服务器和原数据库环境中提高总 cap，collector 会跳过已有 SQL 并继续采集。例如某个普通库原 cap 为 5,000：

```bash
export DB=credit
export NEW_RAW_CAP=5500
export RAW_TARGET="$RAW_ROOT/$DB/complex_workload_200k_s1.json"

cd "$SRC"
"$PY" run_benchmark.py --run_workload \
  --source "$LCM_ROOT/data/workloads/training/$DB/complex_workload_200k_s1.sql" \
  --db_name "$DB" \
  --target "$RAW_TARGET" \
  --database_conn user=postgres,host=localhost \
  --mode raw \
  --query_timeout 30 \
  --repetitions_per_query 1 \
  --min_query_ms 100 \
  --cap_workload "$NEW_RAW_CAP"
```

补采后只重跑该库的第 7.2 节转换。若仍不足，继续把 `NEW_RAW_CAP` 每次增加 200～500。对于 IMDB/TPC-H-PK，不使用这个普通 raw 命令，应回到 `COLLECTION_GUIDE.md` 的 paired 命令同时续采 raw/JSON。


### 7.4 生成统一的 combined feature statistics

```bash
export COMBINED_STATS="$STANDARD_ROOT/statistics_complex_workload_paired_v2.json"

STANDARD_RUNS=()
for db in "${ALL_DBS[@]}"; do
  STANDARD_RUNS+=("$STANDARD_ROOT/$db/complex_workload_200k_s1.json")
done

cd "$SRC"
PYTHONHASHSEED=0 "$PY" - "$COMBINED_STATS" "${STANDARD_RUNS[@]}" <<'PY'
import sys
from training.preprocessing.feature_statistics import gather_feature_statistics

target, *runs = sys.argv[1:]
if len(runs) != 20:
    raise SystemExit(f"需要 20 个 standard parsed 文件，实际得到 {len(runs)} 个")
gather_feature_statistics(runs, target)
print("saved", target)
PY
```

作用：从 20 库实际计划生成一份统一的类别字典和数值缩放统计。Zero-Shot 和 DACE 的所有目标库、所有 seed 都必须复用这一份文件，生成后不要中途重建。

这里包含目标库的 stats，但目标库计划不会进入模型参数训练。原因是当前 Zero-Shot 实现没有 unknown-category 编码；如果 stats 只看其余 19 库，目标库出现新的算子或数据类型时会直接报错。这也是项目训练脚本使用 combined stats 的方式。

### 7.5 选择目标库并构造 19/1 输入

先训练 IMDB：

```bash
export TARGET_DB=imdb
```

训练完 IMDB 后，改为：

```bash
export TARGET_DB=tpc_h_pk
```

每次选好目标库后执行：

```bash
TRAIN_RUNS=()
TEST_RUN=""
for db in "${ALL_DBS[@]}"; do
  run="$STANDARD_ROOT/$db/complex_workload_200k_s1.json"
  if [[ "$db" == "$TARGET_DB" ]]; then
    TEST_RUN="$run"
  else
    TRAIN_RUNS+=("$run")
  fi
done

if [[ "${#TRAIN_RUNS[@]}" -ne 19 || ! -s "$TEST_RUN" ]]; then
  echo "19/1 输入构造失败，请不要启动训练"
fi
echo "train databases: ${#TRAIN_RUNS[@]}"
echo "test workload: $TEST_RUN"
```

作用：自动排除当前目标库，避免手工书写 19 个路径时把目标库误放进训练集。

### 7.6 训练 Zero-Shot

以当前 `TARGET_DB`、seed 0 为例：

```bash
export SEED=0
export ZS_HPARAM="$SRC/conf/zeroshot_hyperparameters/tune_est_best_config.json"

export ZS_MODEL_DIR="$MODEL_ROOT/cross_db_paired/zeroshot/$TARGET_DB"
export ZS_EVAL_DIR="$EVAL_ROOT/cross_db_paired/zeroshot/$TARGET_DB"
mkdir -p "$ZS_MODEL_DIR" "$ZS_EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py \
  --mode train \
  --model_type zeroshot \
  --device cpu \
  --model_dir "$ZS_MODEL_DIR" \
  --target_dir "$ZS_EVAL_DIR" \
  --statistics_file "$COMBINED_STATS" \
  --hyperparameter_path "$ZS_HPARAM" \
  --workload_runs "${TRAIN_RUNS[@]}" \
  --test_workload_runs "$TEST_RUN" \
  --num_workers 0 \
  --seed "$SEED" \
  2>&1 | tee "$ZS_EVAL_DIR/train_seed_${SEED}.log"
```

作用：将 19 库合并后按 80%/20% 分成训练集和验证集；训练结束后，在目标库整个 workload 上测试。Zero-Shot 必须显式使用仓库中实际存在的 `conf/zeroshot_hyperparameters/tune_est_best_config.json`。

### 7.7 检查并训练 DACE

DACE 默认配置假设 stats 中恰好有 20 种 `op_name`，即每个节点是“20 维算子 one-hot + `est_cost` + `est_card`”，所以默认 `node_length=22`；同时默认每个计划最多 22 个节点。先检查本次数据：

```bash
cd "$SRC"
"$PY" - "$COMBINED_STATS" "${STANDARD_RUNS[@]}" <<'PY'
import json
import sys

stats_path, *runs = sys.argv[1:]
with open(stats_path) as file:
    stats = json.load(file)

def node_count(node):
    return 1 + sum(node_count(child) for child in node.get("children", []))

max_nodes = 0
for path in runs:
    with open(path) as file:
        plans = json.load(file).get("parsed_plans", [])
    max_nodes = max(max_nodes, *(node_count(plan) for plan in plans))

op_count = stats["op_name"]["no_vals"]
print("op_name categories:", op_count)
print("required DACE node_length:", op_count + 2)
print("maximum plan nodes:", max_nodes)
print("required DACE pad_length:", max(22, max_nodes))
PY
```

检查打印结果：

- 如果 `required DACE node_length` 是 22 且 `required DACE pad_length` 是 22，可直接使用默认配置。
- 如果不是 22，必须先在 `src/classes/classes.py` 的 `DACEModelConfig` 中，将 `node_length` 和 `pad_length` 分别改成命令打印的值。两个目标库和三个 seed 必须使用完全相同的数值。
- 不预设新数据最终有多少种算子，必须以这段检查命令的实际输出为准。

修改后可以用下面的命令确认代码实际采用的值：

```bash
cd "$SRC"
"$PY" -c 'from classes.classes import DACEModelConfig; c=DACEModelConfig(); print("node_length:", c.node_length, "pad_length:", c.pad_length)'
```

确认配置后训练：

```bash
export SEED=0

export DACE_MODEL_DIR="$MODEL_ROOT/cross_db_paired/dace/$TARGET_DB"
export DACE_EVAL_DIR="$EVAL_ROOT/cross_db_paired/dace/$TARGET_DB"
mkdir -p "$DACE_MODEL_DIR" "$DACE_EVAL_DIR"

cd "$SRC"
set -o pipefail
"$PY" main.py \
  --mode train \
  --model_type dace \
  --device cpu \
  --model_dir "$DACE_MODEL_DIR" \
  --target_dir "$DACE_EVAL_DIR" \
  --statistics_file "$COMBINED_STATS" \
  --workload_runs "${TRAIN_RUNS[@]}" \
  --test_workload_runs "$TEST_RUN" \
  --num_workers 0 \
  --seed "$SEED" \
  2>&1 | tee "$DACE_EVAL_DIR/train_seed_${SEED}.log"
```

作用：与 Zero-Shot 使用相同的 19 库训练、目标库整库测试口径，但使用 DACE 模型。

### 7.8 训练顺序和产物

先分别跑通 `imdb seed 0` 和 `tpc_h_pk seed 0`，确认测试 CSV 正常生成后，再跑 seed 1、2。每次更换目标库都要重新执行第 7.5 节，重新构造对应的 `TRAIN_RUNS` 和 `TEST_RUN`。

主要产物：

```text
data/models/cross_db_paired/zeroshot/<target_db>/zeroshot_<seed>.pt
data/models/cross_db_paired/dace/<target_db>/dace_<seed>.pt
data/evaluation/cross_db_paired/zeroshot/<target_db>/complex_workload_200k_s1_<seed>_test_pred.csv
data/evaluation/cross_db_paired/zeroshot/<target_db>/complex_workload_200k_s1_<seed>_test_stats.csv
data/evaluation/cross_db_paired/dace/<target_db>/complex_workload_200k_s1_<seed>_test_pred.csv
data/evaluation/cross_db_paired/dace/<target_db>/complex_workload_200k_s1_<seed>_test_stats.csv
```

## 8. 如何确认训练成功

查看 checkpoint：

```bash
find "$MODEL_ROOT" -type f -name '*.pt' -print | sort
```

作用：列出已保存的模型。每个模型/数据库/seed 应该有一个 `.pt` 文件。

查看内部测试结果：

```bash
find "$EVAL_ROOT" -type f -name '*_test_*.csv' -print | sort
```

作用：列出训练结束后自动生成的测试 CSV。

- `*_test_pred.csv`：每条查询的真实运行时间、预测时间和 Q-error。
- `*_test_stats.csv`：RMSE、MAPE、Q-error percentile 等汇总结果。
- `train_seed_*.log`：训练过程和报错信息。

`main.py --mode train` 会在训练结束后自动测试，因此不需要再单独执行 `--mode predict`。只有将来收集了独立 evaluation workload，才需要再用 predict。


## 9. 结果口径与执行顺序

### 9.1 当前可以直接比较的结果

```text
单库 native：报告 E2E/QueryFormer 全 master 结果，以及 QPP-Net coverage 后的自身结果
单库 strict matched：三个单库模型在完全相同 query ID 上重新训练和测试
跨库 19/1：Zero-Shot/DACE 使用相同的 20 库 standard parsed 口径
```

Zero-Shot/DACE 的原始 19/1 命令测试目标库整个 standard workload，不等于单库固定 split 的 1,000 条 test，因此不能把它叫作五模型 strict matched。若后续要求五个模型测试 query ID 完全一致，还需要继续修改 workload-agnostic test loader；本指南没有把尚未实现的能力写成已实现。

### 9.2 推荐执行顺序

1. 对 IMDB 执行 paired 收集和 master 检查；
2. 执行 IMDB seed 0 的 native 与 strict matched 单库训练；
3. 对 TPC-H-PK 重复以上步骤；
4. seed 0 全部跑通后再执行 seed 1、2；
5. 对 20 库执行 standard parse 数量审核；
6. 只增量补采 standard parsed 不足的旧库；
7. 固定 combined feature statistics；
8. 分别以 IMDB、TPC-H-PK 为目标执行 Zero-Shot/DACE 19/1 训练；
9. 分开汇总单库 native、单库 matched、跨库 19/1 三类结果。
