# 新数据 CPU 训练指南

本指南只解决一件事：使用 CPU 训练机上已经收集好的计划数据，先训练 E2E、QueryFormer 和 QPP-Net；20 个数据库收齐后，再训练 Zero-Shot 和 DACE，并查看测试结果。

本指南默认仓库位于 `/root/lcm-eval`。如果实际路径不同，只需修改下面的 `LCM_ROOT`。

## 1. 当前分工和输入


| 负责人 | 模型               | 需要的采集数据                   | 训练输入                            | 测试方式            |
| --- | ---------------- | ------------------------- | ------------------------------- | --------------- |
| 房子珈 | E2E              | raw 文本计划                  | baseline parsed + sample bitmap | 单库内部 80/10/10   |
| 李昊森 | QueryFormer      | raw 文本计划                  | baseline parsed + sample bitmap | 单库内部 80/10/10   |
| 何沅东 | QPP-Net          | `--mode json` 采集的 JSON 计划 | 清洗后的 JSON `query_list`          | 单库内部 80/10/10   |
| 暂停  | Zero-Shot / DACE | 20 个数据库的 raw 计划           | standard parsed                 | 其余 19 库训练，目标库测试 |


赵嘉祺负责的以下 6 个数据库还没有收集完：

```text
credit employee fhnk financial geneea genome
```

因此，现在不进行 Zero-Shot 和 DACE 的正式复现。不要把少于 20 库的实验当成论文的 19 库训练/1 库测试结果。

本次实验已确定只在下面两个目标库上训练和测试 workload-driven 模型：

```text
imdb tpc_h_pk
```

所以 E2E、QueryFormer 和 QPP-Net 都只需要在 `imdb` 和 `tpc_h_pk` 上分别训练。

## 2. 训练机需要哪些文件

E2E 和 QueryFormer 需要：

```text
/root/lcm-eval/data/raw_complex/imdb/complex_workload_200k_s1.json
/root/lcm-eval/data/raw_complex/tpc_h_pk/complex_workload_200k_s1.json

/root/lcm-eval/data/datasets/imdb/*.csv
/root/lcm-eval/data/datasets/tpc_h/*.csv
```

QPP-Net 需要：

```text
/root/lcm-eval/data/json_complex/imdb/complex_workload_200k_s1/complex_workload_200k_s1.json
/root/lcm-eval/data/json_complex/tpc_h_pk/complex_workload_200k_s1/complex_workload_200k_s1.json
```

开始预处理前，只需要检查上述路径中的数据是否已经存在：

```bash
for db in imdb tpc_h_pk; do
  raw="/root/lcm-eval/data/raw_complex/$db/complex_workload_200k_s1.json"
  json="/root/lcm-eval/data/json_complex/$db/complex_workload_200k_s1/complex_workload_200k_s1.json"
  if [[ "$db" == tpc_h_pk ]]; then
    csv_dir="/root/lcm-eval/data/datasets/tpc_h"
  else
    csv_dir="/root/lcm-eval/data/datasets/$db"
  fi

  [[ -s "$raw" ]] \
    && echo "OK      $raw" \
    || echo "MISSING $raw"

  [[ -s "$json" ]] \
    && echo "OK      $json" \
    || echo "MISSING $json"

  if [[ -d "$csv_dir" ]] && find "$csv_dir" -maxdepth 1 -type f -name '*.csv' -print -quit | grep -q .; then
    echo "OK      $csv_dir/*.csv"
  else
    echo "MISSING $csv_dir/*.csv"
  fi
done
```

作用：只读检查 IMDB 和 TPC-H-PK 的 raw、JSON mode 计划以及 CSV 数据是否存在，不会复制或修改任何数据。全部显示 `OK` 后再继续；出现 `MISSING` 时，请先让对应数据负责人补齐该路径。

训练机不需要运行 PostgreSQL，也不需要重新执行 SQL。

## 3. 一次性 CPU 环境

### 3.1 已有 Python 环境时

如果训练机已经有可用的 `.venv`，先执行：

```bash
cd /root/lcm-eval/src
/root/lcm-eval/.venv/bin/python -c 'import torch, dgl, gensim; print("environment OK")'
```

作用：检查 PyTorch、DGL 和 word2vec 依赖能否导入。如果打印 `environment OK`，可以跳过下一节。项目的 `main.py` 在第 3.3 节设置好 `NODE00...NODE05` 后再检查。

### 3.2 没有环境时

 参考 [https://github.com/fzj2007/LCM-Reproduction-Script](https://github.com/fzj2007/LCM-Reproduction-Script) 已验证过 CPU 版 PyTorch 2.3.0 + DGL 1.1.3。在 `/root/lcm-eval` 中建立独立环境：

```bash
cd /root/lcm-eval
python3 -m venv .venv-cpu

export PY=/root/lcm-eval/.venv-cpu/bin/python

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
export LCM_ROOT=/root/lcm-eval
export SRC="$LCM_ROOT/src"

# 仓库中已经有 .venv 时使用这一行：
export PY="$LCM_ROOT/.venv/bin/python"

# 只有按第 3.2 节新建了 .venv-cpu 时，才注释上一行并改用这一行：
# export PY="$LCM_ROOT/.venv-cpu/bin/python"

export RAW_ROOT="$LCM_ROOT/data/raw_complex"
export JSON_ROOT="$LCM_ROOT/data/json_complex"
export BASELINE_ROOT="$LCM_ROOT/data/parsed_complex_baseline"
export AUG_ROOT="$LCM_ROOT/data/augmented_complex_baseline"
export STANDARD_ROOT="$LCM_ROOT/data/parsed_complex"
export SENT_ROOT="$LCM_ROOT/data/sentences"
export CLEAN_JSON_ROOT="$LCM_ROOT/data/json_complex_clean"
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

mkdir -p "$BASELINE_ROOT" "$AUG_ROOT" "$SENT_ROOT"
mkdir -p "$CLEAN_JSON_ROOT" "$STATS_ROOT" "$MODEL_ROOT" "$EVAL_ROOT"

cd "$SRC"
"$PY" -c 'import main; print("main.py OK")'
```

最后打印 `main.py OK` 即表示环境可以用。

## 4. E2E 和 QueryFormer 的共享预处理

E2E 和 QueryFormer 使用相同的 augmented 训练数据。这一节每个数据库只需执行一次；建议由房子珈生成，李昊森直接复用。

先选择数据库。每次只执行下面一组：

```bash
# IMDB
export TARGET_DB=imdb
export CSV_DB=imdb
export CAP=10000

# TPC-H-PK（跑 TPC-H-PK 时改用这三行）
# export TARGET_DB=tpc_h_pk
# export CSV_DB=tpc_h
# export CAP=10000
```

`TARGET_DB` 是计划和仓库元数据使用的数据库名；`CSV_DB` 是实际 CSV 目录名。TPC-H-PK 的计划目录叫 `tpc_h_pk`，但它复用 `data/datasets/tpc_h` 下的 CSV。

选好后，依次执行下面 5 步。

### 4.1 raw → baseline parsed

```bash
export RAW_RUN="$RAW_ROOT/$TARGET_DB/complex_workload_200k_s1.json"
export BASELINE_RUN="$BASELINE_ROOT/$TARGET_DB/complex_workload_200k_s1.json"

mkdir -p "$(dirname "$BASELINE_RUN")"

cd "$SRC"
"$PY" run_benchmark.py \
  --parse_run \
  --source "$RAW_RUN" \
  --target "$BASELINE_RUN" \
  --parse_baseline \
  --min_query_ms 100 \
  --max_query_ms 30000 \
  --cap_queries "$CAP"
```

作用：读取 raw `query_list`，去掉无效计划以及运行时间不在 100ms～30000ms 的计划，生成 baseline parsed。

`--cap_queries 10000` 表示 baseline parsed 最多输出 10000 条。目前实测 JOB（IMDB）虽然采集到了 10000 条有效 raw 查询，但转换时会跳过其中 13 条包含 `InitPlan` 的查询，因此当前只得到 9987 条 baseline parsed。为了保证 JOB 的最终训练输入为 10000 条，需要继续补充采集新的有效查询，然后重新转换，并以 baseline parsed 实际达到 10000 条为准。TPC-H-PK 实测不需要补采，现有 10000 条 raw 已全部成功转换为 10000 条 baseline parsed。

转换中出现的 `did not find enough filters` 警告不会删除整条计划，但表示该计划中的个别复杂 `OR/IN` 过滤条件没有被解析器完整展开。

转换后检查实际条数：

```bash
"$PY" - "$BASELINE_RUN" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    count = len(json.load(file).get("parsed_plans", []))
print("baseline parsed:", count)
if count < 10000:
    raise SystemExit("不足 10000 条：请补采 raw 查询后重新转换")
PY
```

只有打印 `baseline parsed: 10000` 后，才继续生成 sample bitmap。JOB 当前的 9987 条只用于跑通流程，正式训练前仍需补齐。

产物：`data/parsed_complex_baseline/<db>/complex_workload_200k_s1.json`。

### 4.2 增加 sample bitmap

```bash
export AUG_RUN="$AUG_ROOT/$TARGET_DB/complex_workload_200k_s1.json"
export CSV_DIR="$LCM_ROOT/data/datasets/$CSV_DB"

cd "$SRC"
"$PY" - "$TARGET_DB" "$CSV_DIR" \
  "$BASELINE_RUN" "$AUG_RUN" <<'PY'
import sys
from models.workload_driven.preprocessing.sample_vectors import augment_sample_vectors

augment_sample_vectors(
    dataset=sys.argv[1],
    data_dir=sys.argv[2],
    plan_path=sys.argv[3],
    target_path=sys.argv[4],
)
PY
```

作用：读取 `data/datasets/<db>/*.csv`，为计划中的谓词生成 sample bitmap。

产物：`data/augmented_complex_baseline/<db>/complex_workload_200k_s1.json`。

如果目标文件已存在，该函数会直接跳过。如果 raw 重新收集过，请不要沿用旧 augmented 文件。

### 4.3 生成 sentences

```bash
export SENTENCES="$SENT_ROOT/$TARGET_DB/sentences.json"

cd "$SRC"
"$PY" - "$TARGET_DB" "$BASELINE_RUN" \
  "$CSV_DIR" "$SENTENCES" <<'PY'
import sys
from models.workload_driven.preprocessing.sentence_creation import create_sentences

create_sentences(
    dataset=sys.argv[1],
    plan_paths=[sys.argv[2]],
    data_dir=sys.argv[3],
    target=sys.argv[4],
)
PY
```

作用：从数据库 CSV 和查询谓词中提取用于训练 word2vec 的句子。

产物：`data/sentences/<db>/sentences.json`。

### 4.4 生成 word2vec

```bash
export WORD2VEC="$SENT_ROOT/$TARGET_DB/word2vec.m"

cd "$SRC"
"$PY" - "$SENTENCES" "$WORD2VEC" <<'PY'
import sys
from models.workload_driven.preprocessing.word_embeddings import compute_word_embeddings

compute_word_embeddings(sys.argv[1], sys.argv[2])
PY
```

作用：用 sentences 训练字符串词向量。E2E 和 QueryFormer 训练都会读取该文件。

产物：`data/sentences/<db>/word2vec.m`。

同时生成的 `word2vec.m.vectors.npy` 也是词向量文件的一部分，请保留。

该函数默认使用 `CPU 核数 - 2` 个 worker，因此机器至少需要 3 个逻辑 CPU。

### 4.5 生成 feature statistics

```bash
export AUG_STATS="$STATS_ROOT/augmented/$TARGET_DB/feature_statistics.json"

cd "$SRC"
"$PY" - "$AUG_RUN" "$AUG_STATS" <<'PY'
import sys
from training.preprocessing.feature_statistics import gather_feature_statistics

gather_feature_statistics([sys.argv[1]], sys.argv[2])
print("saved", sys.argv[2])
PY
```

作用：统计目标库的数值范围和类别字典，供模型进行特征缩放和编码。

产物：`data/feature_statistics/augmented/<db>/feature_statistics.json`。

当前操作很简单：

- 训练 IMDB 模型时，使用 IMDB 生成的 `feature_statistics.json`。
- 训练 TPC-H-PK 模型时，使用 TPC-H-PK 生成的 `feature_statistics.json`。
- 因此，其他 6 个还没收集完的数据库不会影响当前 E2E 和 QueryFormer 训练。

`feature_statistics.json` 只是对训练数据中的数值范围和类别进行统计，不是另外一份查询训练数据。

## 5. 房子珈：训练 E2E

先确保第 4 节已经为当前 `TARGET_DB` 生成 `AUG_RUN`、`AUG_STATS` 和 `WORD2VEC`。

以 IMDB、seed 0 为例：

```bash
export TARGET_DB=imdb
export SEED=0

export AUG_RUN="$AUG_ROOT/$TARGET_DB/complex_workload_200k_s1.json"
export AUG_STATS="$STATS_ROOT/augmented/$TARGET_DB/feature_statistics.json"
export WORD2VEC="$SENT_ROOT/$TARGET_DB/word2vec.m"
export COLUMN_STATS="$SRC/cross_db_benchmark/datasets/$TARGET_DB/column_statistics.json"

mkdir -p "$MODEL_ROOT/e2e/$TARGET_DB" "$EVAL_ROOT/e2e/$TARGET_DB"

cd "$SRC"
set -o pipefail
"$PY" main.py \
  --mode train \
  --model_type e2e \
  --device cpu \
  --model_dir "$MODEL_ROOT/e2e/$TARGET_DB" \
  --target_dir "$EVAL_ROOT/e2e/$TARGET_DB" \
  --statistics_file "$AUG_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$AUG_RUN" \
  --num_workers 0 \
  --seed "$SEED" \
  2>&1 | tee "$EVAL_ROOT/e2e/$TARGET_DB/train_seed_${SEED}.log"
```

作用：使用这一个数据库的 augmented workload 训练 E2E。程序自动按 80% train、10% validation、10% test 切分，训练结束后自动测试，所以这条命令不需要 `--test_workload_runs`。

主要产物：

```text
data/models/e2e/<db>/e2e_0.pt
data/evaluation/e2e/<db>/complex_workload_200k_s1_0_test_pred.csv
data/evaluation/e2e/<db>/complex_workload_200k_s1_0_test_stats.csv
```

IMDB 跑通后，把 `TARGET_DB` 改为 `tpc_h_pk` 再跑一次。两库的 seed 0 都成功后，再依次使用 seed 1、2。

## 6. 李昊森：训练 QueryFormer

QueryFormer 直接复用第 4 节产生的数据。以 IMDB、seed 0 为例：

```bash
export TARGET_DB=imdb
export SEED=0

export AUG_RUN="$AUG_ROOT/$TARGET_DB/complex_workload_200k_s1.json"
export AUG_STATS="$STATS_ROOT/augmented/$TARGET_DB/feature_statistics.json"
export WORD2VEC="$SENT_ROOT/$TARGET_DB/word2vec.m"
export COLUMN_STATS="$SRC/cross_db_benchmark/datasets/$TARGET_DB/column_statistics.json"

mkdir -p "$MODEL_ROOT/query_former/$TARGET_DB" "$EVAL_ROOT/query_former/$TARGET_DB"

cd "$SRC"
set -o pipefail
"$PY" main.py \
  --mode train \
  --model_type query_former \
  --device cpu \
  --model_dir "$MODEL_ROOT/query_former/$TARGET_DB" \
  --target_dir "$EVAL_ROOT/query_former/$TARGET_DB" \
  --statistics_file "$AUG_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$AUG_RUN" \
  --num_workers 0 \
  --seed "$SEED" \
  2>&1 | tee "$EVAL_ROOT/query_former/$TARGET_DB/train_seed_${SEED}.log"
```

作用：使用目标库 augmented workload 训练 QueryFormer。它也会自动按 80/10/10 切分并在训练结束后测试。

主要产物：

```text
data/models/query_former/<db>/query_former_0.pt
data/evaluation/query_former/<db>/complex_workload_200k_s1_0_test_pred.csv
data/evaluation/query_former/<db>/complex_workload_200k_s1_0_test_stats.csv
```

同样在 `imdb`、`tpc_h_pk` 上分别训练，先 seed 0，再 seed 1、2。

## 7. 何沅东：准备并训练 QPP-Net

QPP-Net 不使用第 4 节的 baseline parsed、sample bitmap 和 word2vec。它直接使用 PostgreSQL JSON mode 计划。

先选择目标库，以 IMDB 为例：

```bash
export QPP_DB=imdb
export QPP_SOURCE="$JSON_ROOT/$QPP_DB/complex_workload_200k_s1/complex_workload_200k_s1.json"
export QPP_CLEAN="$CLEAN_JSON_ROOT/$QPP_DB/complex_workload_200k_s1.json"
```

### 7.1 清洗 JSON 训练数据

```bash
cd "$SRC"
"$PY" - "$QPP_SOURCE" "$QPP_CLEAN" <<'PY'
import copy
import json
import os
import sys
from collections import Counter

from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from cross_db_benchmark.benchmark_tools.postgres.check_valid import check_valid
from training.featurizations import QPPNetFeaturization

source, target = sys.argv[1:3]
with open(source) as file:
    run = json.load(file)

supported = set(QPPNetFeaturization.QPP_NET_OPERATOR_TYPES)
join_types = {"Hash Join", "Merge Join", "Nested Loop"}

def unsupported_nodes(node):
    node_type = node.get("Node Type")
    normalized = "Join" if node_type in join_types else node_type
    result = [] if normalized in supported else [str(node_type)]
    for child in node.get("Plans", []) or []:
        result.extend(unsupported_nodes(child))
    return result

clean = []
unsupported = Counter()
for query in run.get("query_list", []):
    if not check_valid(ExecutionMode.JSON_OUTPUT, query, min_runtime=100, verbose=False):
        continue
    analyze = query["analyze_plans"][0]
    if float(analyze.get("Execution Time", 0)) > 30000:
        continue
    bad = unsupported_nodes(analyze["Plan"])
    if bad:
        unsupported.update(bad)
        continue
    clean.append(query)

if not clean:
    raise SystemExit("清洗后没有可用查询")

result = copy.deepcopy(run)
result["query_list"] = clean
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, "w") as file:
    json.dump(result, file)

print("raw queries:", len(run.get("query_list", [])))
print("clean queries:", len(clean))
print("filtered unsupported operators:", dict(unsupported))
print("saved:", target)
PY
```

作用：去掉 SQL 执行错误、超时、空计划、零基数、小于 100ms、大于 30000ms 的查询，并过滤当前 QPP-Net 不支持的算子。

产物：`data/json_complex_clean/<db>/complex_workload_200k_s1.json`。

### 7.2 生成 QPP-Net feature statistics

```bash
export QPP_STATS="$STATS_ROOT/qppnet/$QPP_DB/feature_statistics.json"

cd "$SRC"
"$PY" - "$QPP_CLEAN" "$QPP_STATS" <<'PY'
import sys
from training.preprocessing.feature_statistics import gather_feature_statistics

gather_feature_statistics([sys.argv[1]], sys.argv[2])
print("saved", sys.argv[2])
PY
```

作用：从清洗后的 JSON 计划生成 QPP-Net 的数值缩放和类别编码信息。

### 7.3 训练 QPP-Net

```bash
export SEED=0
export COLUMN_STATS="$SRC/cross_db_benchmark/datasets/$QPP_DB/column_statistics.json"

mkdir -p "$MODEL_ROOT/qppnet/$QPP_DB" "$EVAL_ROOT/qppnet/$QPP_DB"

cd "$SRC"
set -o pipefail
"$PY" main.py \
  --mode train \
  --model_type qppnet \
  --device cpu \
  --model_dir "$MODEL_ROOT/qppnet/$QPP_DB" \
  --target_dir "$EVAL_ROOT/qppnet/$QPP_DB" \
  --statistics_file "$QPP_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --workload_runs "$QPP_CLEAN" \
  --num_workers 0 \
  --seed "$SEED" \
  2>&1 | tee "$EVAL_ROOT/qppnet/$QPP_DB/train_seed_${SEED}.log"
```

作用：在当前数据库的 clean JSON 上训练 QPP-Net，并自动执行 80/10/10 切分和内部测试。

主要产物：

```text
data/models/qppnet/<db>/qppnet_0.pt
data/evaluation/qppnet/<db>/complex_workload_200k_s1_0_test_pred.csv
data/evaluation/qppnet/<db>/complex_workload_200k_s1_0_test_stats.csv
```

完成 IMDB 后，把 `QPP_DB` 改为 `tpc_h_pk`，重复第 7.1～7.3 节。先 seed 0，再 seed 1、2。

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

`main.py --mode train` 已自动测试，所以第一轮不需要再单独执行 `--mode predict`。只有将来收集了独立 evaluation workload，才需要再用 predict。

## 9. 后续训练 Zero-Shot 和 DACE

Zero-Shot 和 DACE 是 workload-agnostic 模型。对每个目标库，它们使用其余 19 库训练，并把目标库的整个 workload 作为测试集。它们只读取 standard parsed，不需要 sample bitmap、CSV 或 word2vec。

目前 6 个数据库还没有收集完成，因此本节命令留到 20 库全部收齐后执行。

### 9.1 检查 20 库 raw

```bash
ALL_DBS=(
  accidents airline baseball basketball carcinogenesis consumer
  credit employee fhnk financial geneea genome hepatitis imdb
  movielens seznam ssb tournament tpc_h_pk walmart
)

missing=0
for db in "${ALL_DBS[@]}"; do
  file="$RAW_ROOT/$db/complex_workload_200k_s1.json"
  if [[ -s "$file" ]]; then
    echo "OK      $db"
  else
    echo "MISSING $db"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "20 库未收齐，请不要继续第 9.2 节"
fi
```

作用：只读检查官方列表中的 20 个 raw 文件。必须全部显示 `OK`。

### 9.2 生成 20 库 standard parsed

```bash
mkdir -p "$STANDARD_ROOT"

for db in "${ALL_DBS[@]}"; do
  expected=5000
  if [[ "$db" == imdb || "$db" == tpc_h_pk ]]; then
    expected=10000
  fi

  source="$RAW_ROOT/$db/complex_workload_200k_s1.json"
  target="$STANDARD_ROOT/$db/complex_workload_200k_s1.json"
  mkdir -p "$(dirname "$target")"

  cd "$SRC"
  "$PY" run_benchmark.py \
    --parse_run \
    --source "$source" \
    --target "$target" \
    --parse_join_conds \
    --min_query_ms 100 \
    --max_query_ms 30000 \
    --cap_queries "$expected"

  count=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["parsed_plans"]))' "$target")
  echo "$db: $count/$expected"
  if [[ "$count" -ne "$expected" ]]; then
    echo "$db 转换后不足 $expected 条，请数据负责人补采后重新转换"
    break
  fi
done
```

作用：将 20 库 raw 转成 Zero-Shot/DACE 共用的 standard parsed。这里没有 `--parse_baseline`；不要使用 E2E/QueryFormer 的 `parsed_complex_baseline` 代替。

数量要求：IMDB、TPC-H-PK 各 10000 条，其他 18 库各 5000 条。这样无论以 IMDB 还是 TPC-H-PK 为目标，剩余 19 库都正好提供约 100000 条训练计划。如果某库因 `SubPlan/InitPlan` 被跳过而不足目标数量，应先补采，不能继续训练。

### 9.3 生成统一的 combined feature statistics

```bash
export COMBINED_STATS="$STANDARD_ROOT/statistics_complex_workload_combined.json"

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

### 9.4 选择目标库并构造 19/1 输入

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

### 9.5 训练 Zero-Shot

以当前 `TARGET_DB`、seed 0 为例：

```bash
export SEED=0
export ZS_HPARAM="$SRC/conf/zeroshot_hyperparameters/tune_est_best_config.json"

mkdir -p "$MODEL_ROOT/zeroshot/$TARGET_DB" "$EVAL_ROOT/zeroshot/$TARGET_DB"

cd "$SRC"
set -o pipefail
"$PY" main.py \
  --mode train \
  --model_type zeroshot \
  --device cpu \
  --model_dir "$MODEL_ROOT/zeroshot/$TARGET_DB" \
  --target_dir "$EVAL_ROOT/zeroshot/$TARGET_DB" \
  --statistics_file "$COMBINED_STATS" \
  --hyperparameter_path "$ZS_HPARAM" \
  --workload_runs "${TRAIN_RUNS[@]}" \
  --test_workload_runs "$TEST_RUN" \
  --num_workers 0 \
  --seed "$SEED" \
  2>&1 | tee "$EVAL_ROOT/zeroshot/$TARGET_DB/train_seed_${SEED}.log"
```

作用：将 19 库合并后按 80%/20% 分成训练集和验证集；训练结束后，在目标库整个 workload 上测试。Zero-Shot 必须显式使用仓库中实际存在的 `conf/zeroshot_hyperparameters/tune_est_best_config.json`。

### 9.6 检查并训练 DACE

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
- 当前 IMDB 和 TPC-H-PK 已经观察到 21 种算子，因此最终 combined stats 很可能要求 `node_length=23`。必须以 20 库全部转换后的检查结果为准。该修改是为了让 DACE 兼容本次新数据；如果坚持论文旧环境的默认结构，则应保留 22 并停止本次 DACE 实验，不能用错位特征训练。

修改后可以用下面的命令确认代码实际采用的值：

```bash
cd "$SRC"
"$PY" -c 'from classes.classes import DACEModelConfig; c=DACEModelConfig(); print("node_length:", c.node_length, "pad_length:", c.pad_length)'
```

确认配置后训练：

```bash
export SEED=0

mkdir -p "$MODEL_ROOT/dace/$TARGET_DB" "$EVAL_ROOT/dace/$TARGET_DB"

cd "$SRC"
set -o pipefail
"$PY" main.py \
  --mode train \
  --model_type dace \
  --device cpu \
  --model_dir "$MODEL_ROOT/dace/$TARGET_DB" \
  --target_dir "$EVAL_ROOT/dace/$TARGET_DB" \
  --statistics_file "$COMBINED_STATS" \
  --workload_runs "${TRAIN_RUNS[@]}" \
  --test_workload_runs "$TEST_RUN" \
  --num_workers 0 \
  --seed "$SEED" \
  2>&1 | tee "$EVAL_ROOT/dace/$TARGET_DB/train_seed_${SEED}.log"
```

作用：与 Zero-Shot 使用相同的 19 库训练、目标库整库测试口径，但使用 DACE 模型。

### 9.7 训练顺序和产物

先分别跑通 `imdb seed 0` 和 `tpc_h_pk seed 0`，确认测试 CSV 正常生成后，再跑 seed 1、2。每次更换目标库都要重新执行第 9.4 节，重新构造对应的 `TRAIN_RUNS` 和 `TEST_RUN`。

主要产物：

```text
data/models/zeroshot/<target_db>/zeroshot_<seed>.pt
data/models/dace/<target_db>/dace_<seed>.pt
data/evaluation/zeroshot/<target_db>/complex_workload_200k_s1_<seed>_test_pred.csv
data/evaluation/zeroshot/<target_db>/complex_workload_200k_s1_<seed>_test_stats.csv
data/evaluation/dace/<target_db>/complex_workload_200k_s1_<seed>_test_pred.csv
data/evaluation/dace/<target_db>/complex_workload_200k_s1_<seed>_test_stats.csv
```

## 10. `LCM-Reproduction-Script` 怎么使用

`LCM-Reproduction-Script` 对本次工作的作用是：

- 参考已验证的 CPU 依赖版本；
- 参考 `main.py` 的训练参数；
- 参考 workload-driven 的 80/10/10 和 workload-agnostic 的 19/1 训练方式。

不要在本次新数据上直接执行：

```bash
bash reproduce.sh train
```

原因是该脚本使用它自己目录下的官方 OSF/c8220 数据，并且固定读取 `workload_100k_s1_c8220.json`，不会自动读取本指南的 `raw_complex`、`json_complex` 和 `complex_workload_200k_s1.json`。

## 11. 为什么没有直接照抄项目 README

项目 README 描述的总流程是对的：解析数据、生成 sample bitmap、生成 feature statistics、调用 `main.py`。但当前代码与 README 有几个不一致：

- README 中的 `gather_feature_statistics.py` 文件不存在；
- `gather_feature_stats.py` 只会拼接旧 `data/runs/json` 目录；
- `baseline.py` 启动时会导入一个不存在的模块；
- `parse_all.py` 当前把 `min_query_ms` 写死为 0。

本次实际解析还修正了 `parse_filter.py` 对新版本 PostgreSQL `= ANY (...)` 表达式的识别问题，并通过了现有 filter parsing 测试。训练机应使用当前工作区中的同版源码。

所以本指南使用 `run_benchmark.py --parse_run` 和底层 Python 函数，但整体流程仍然与 README 一致。

## 12. 执行顺序总结

1. 所有人执行第 3 节，准备 CPU 环境。
2. 房子珈对 imdb、tpc_h_pk 各执行一次第 4 节共享预处理。
3. 房子珈执行第 5 节，训练 E2E。
4. 李昊森复用共享产物，执行第 6 节，训练 QueryFormer。
5. 何沅东对 imdb、tpc_h_pk 分别执行第 7 节，训练 QPP-Net。
6. 所有人用第 8 节的两条 `find` 命令检查 checkpoint 和测试 CSV。
7. 等 6 个缺失数据库全部收齐后，按第 9.1～9.3 节生成 20 库 standard parsed 和 combined stats。
8. 分别以 imdb、tpc_h_pk 为目标库，按第 9.4～9.7 节训练和测试 Zero-Shot、DACE。

建议实验顺序：

```text
imdb seed 0 跑通
  → tpc_h_pk seed 0
  → 检查所有 test CSV
  → seed 1
  → seed 2
```
