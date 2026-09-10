# IMDB / TPC-H-PK 对齐训练指南

本指南训练 E2E、QueryFormer 和 QPP-Net。数据必须先按 `COLLECTION_GUIDE.md` 生成 baseline master、JSON master、alignment manifest 和 split。

## 1. 公共环境

```bash
export LCM_ROOT=/data/workspace/lcm-eval
export SRC="$LCM_ROOT/src"
export PY="$LCM_ROOT/.venv/bin/python"
export PYTHONPATH="$SRC"
export PYTHONHASHSEED=0
export DGLBACKEND=pytorch
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

NODE_ENV='{"hostname":"localhost","python":"3.10"}'
export NODE00="$NODE_ENV" NODE01="$NODE_ENV" NODE02="$NODE_ENV"
export NODE03="$NODE_ENV" NODE04="$NODE_ENV" NODE05="$NODE_ENV"
```

选择数据库和 seed：

```bash
export DB=imdb       # 或 tpc_h_pk
export SEED=0        # 之后运行1、2

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

## 2. 生成公共预处理资源

### 2.1 sentences 和 word2vec

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

### 2.2 feature statistics

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

## 3. 查看 QPP-Net coverage

```bash
cd "$SRC"
"$PY" evaluation/aligned_metrics.py \
  --alignment_manifest "$ALIGNMENT" \
  --split_manifest "$SPLIT"
```

它会分别输出 train、validation、test 中 QPP-Net 支持的数量和比例。

## 4. Native 训练

### 4.1 E2E

```bash
export MODEL_DIR="$LCM_ROOT/data/models/baseline_native/e2e/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/baseline_native/e2e/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
"$PY" main.py --mode train --model_type e2e --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol baseline_native \
  --num_workers 0 --seed "$SEED"
```

### 4.2 QueryFormer

```bash
export MODEL_DIR="$LCM_ROOT/data/models/baseline_native/query_former/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/baseline_native/query_former/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
"$PY" main.py --mode train --model_type query_former --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol baseline_native \
  --num_workers 0 --seed "$SEED"
```

### 4.3 QPP-Net

```bash
export MODEL_DIR="$LCM_ROOT/data/models/qpp_native/qppnet/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/qpp_native/qppnet/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
"$PY" main.py --mode train --model_type qppnet --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$QPP_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --workload_runs "$JSON_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol qpp_native \
  --num_workers 0 --seed "$SEED"
```

Native 结果反映各自原生可用范围；QPP-Net 结果必须连同 coverage 报告，不能直接作为严格 matched 对比。

## 5. Strict matched 训练

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

三个 dataloader 都会在原 master split 内应用同一 QPP support mask。不能复用第 4 节使用完整 baseline train 训练出的 E2E/QueryFormer checkpoint。

### 5.1 Matched E2E

```bash
export MODEL_DIR="$LCM_ROOT/data/models/matched/e2e/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/matched/e2e/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
"$PY" main.py --mode train --model_type e2e --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol matched \
  --num_workers 0 --seed "$SEED"
```

### 5.2 Matched QueryFormer

```bash
export MODEL_DIR="$LCM_ROOT/data/models/matched/query_former/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/matched/query_former/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
"$PY" main.py --mode train --model_type query_former --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$BASELINE_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --word_embeddings "$WORD2VEC" \
  --workload_runs "$BASELINE_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol matched \
  --num_workers 0 --seed "$SEED"
```

### 5.3 Matched QPP-Net

```bash
export MODEL_DIR="$LCM_ROOT/data/models/matched/qppnet/$DB"
export EVAL_DIR="$LCM_ROOT/data/evaluation/matched/qppnet/$DB"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

cd "$SRC"
"$PY" main.py --mode train --model_type qppnet --device cpu \
  --model_dir "$MODEL_DIR" --target_dir "$EVAL_DIR" \
  --statistics_file "$QPP_STATS" \
  --column_statistics "$COLUMN_STATS" \
  --workload_runs "$JSON_MASTER" \
  --split_manifest "$SPLIT" \
  --alignment_manifest "$ALIGNMENT" \
  --experiment_protocol matched \
  --num_workers 0 --seed "$SEED"
```

## 6. 验证 strict matched 结果

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

## 7. 执行顺序

```text
IMDB seed 0 native跑通
→ IMDB seed 0 matched跑通并验证ID
→ TPC-H-PK seed 0重复
→ seed 1、2
→ 汇总三个seed和coverage
```
