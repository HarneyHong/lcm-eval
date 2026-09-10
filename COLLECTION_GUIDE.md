# 数据收集与清洗指南

## 0. 产出（新目录）


| 阶段                    | 目录                                                                | 内容                           |
| --------------------- | ----------------------------------------------------------------- | ---------------------------- |
| raw                   | `data/raw_complex/<db>/complex_workload_200k_s1.json`             | EXPLAIN ANALYZE 文本（5000 条有效） |
| parsed                | `data/parsed_complex/<db>/complex_workload_200k_s1.json`          | 标准解析计划                       |
| baseline              | `data/parsed_complex_baseline/<db>/complex_workload_200k_s1.json` | MSCN/E2E/QueryFormer 用解析     |
| json（仅 imdb/tpc_h_pk） | `data/json_complex/<db>/complex_workload_200k_s1/...`             | QPP-Net 用                    |


## 1. 环境准备

```bash
git clone https://github.com/DataManagementLab/lcm-eval.git
cd lcm-eval
python3 -m venv .venv && .venv/bin/pip install -r requirements/requirements.txt
# 其他：
# 安装 PostgreSQL
# .env 里 LOCAL_ROOT_PATH 指向仓库根目录
```

## 2. 下载数据

```bash
cd src
.venv/bin/python download_from_osf.py --artifacts datasets   # -> data/datasets/<db>/*.csv
.venv/bin/python download_from_osf.py --artifacts workloads  # -> data/workloads/training/<db>/complex_workload_200k_s1.sql
```

## 3. 收集数量约定


| 库             | cap_workload                             |
| ------------- | ---------------------------------------- |
| imdb、tpc_h_pk | **10000**（workload-driven 主目标，对齐论文已发布数据） |
| 其余所有库         | **5000**                                 |


## 4. 收集（每个库两步，每个人收集不同的库）

```bash
cd src
# 灌库
python run_benchmark.py --load_database \
  --data_dir data/datasets --dataset <db> --db_name <db> \
  --database_conn user=postgres,host=localhost

# 收集 raw
python run_benchmark.py --run_workload \
  --source data/workloads/training/<db>/complex_workload_200k_s1.sql --db_name <db> \
  --target data/raw_complex/<db>/complex_workload_200k_s1.json \
  --database_conn user=postgres,host=localhost --mode raw \
  --query_timeout 30 --repetitions_per_query 1 --cap_workload <CAP>
# <CAP> = 10000（imdb/tpc_h_pk）或 5000（其余库）

# QPP-Net 目标库（imdb/baseball/tpc_h_pk）再加 json 版：
python run_benchmark.py --run_workload \
  --source data/workloads/training/<db>/complex_workload_200k_s1.sql --db_name <db> \
  --target data/json_complex/<db>/complex_workload_200k_s1/complex_workload_200k_s1.json \
  --database_conn user=postgres,host=localhost --mode json \
  --query_timeout 30 --repetitions_per_query 1 --cap_workload <CAP>
# json 版用与 raw 相同的 <CAP>
```

## 5. 清洗（主机器统一执行）

```bash
cd src
python parse_all.py \
  --raw_dir data/raw_complex --parsed_plan_dir data/parsed_complex \
  --parsed_plan_dir_baseline data/parsed_complex_baseline \
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
- 完成检查：raw JSON 里 `query_list` 长度 == cap；是否能够直接到LCM模型中训练。
