# How Good Are Learned Cost Models, Really? Insights from Query Optimization Tasks 

> 本工作区的精简入口只公开 End-To-End、QueryFormer、Zero-Shot、DACE 和 QPP-Net；运行方法与 JOB-light 示例见 [FOCUSED_README.md](FOCUSED_README.md)。

This repository contains the evaluation **source code** of the SIGMOD Paper: "[How Good Are Learned Cost Models, Really?
Insights from Query Optimization Tasks](https://dl.acm.org/doi/10.1145/3725309)" from Roman Heinrich, Manisha Luthra,
Johannes Wehrstein, Harald Kornmayer and Carsten Binnig. It provides a set of **Learned Cost Models (LCMs)** that are
trained on different datasets and evaluated on various query optimization tasks.

If you find this paper useful, please cite it as follows:
```bibtex
  @article{10.1145/3725309,
    author = {Heinrich, Roman and Luthra, Manisha and Wehrstein, Johannes and Kornmayer, Harald and Binnig, Carsten},
    title = {How Good are Learned Cost Models, Really? Insights from Query Optimization Tasks},
    year = {2025},
    issue_date = {June 2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {3},
    number = {3},
    url = {https://doi.org/10.1145/3725309},
    doi = {10.1145/3725309},
    journal = {Proc. ACM Manag. Data},
    month = jun,
    articleno = {172},
    numpages = {27},
}
```

# Table of Contents
1. [Overview](#1-overview)  
   1.1 [Overall Workflow](#11-overall-workflow)  
   1.2 [Local and Remote Operations](#12-local-and-remote-operations)  
   1.3 [Hardware Instances](#13-hardware-instances)  
2. [Prerequisites](#2-prerequisites)  
   2.1 [Python environment](#21-python-environment)  
   2.2 [Check environment variables](#22-check-environment-variables)  
   2.3 [Download precollected artifacts](#23-download-precollected-artifacts)  
3. [Query Execution](#3query-execution)  
   3.1 [Downloading query workloads](#31-downloading-query-workloads)  
   3.2 [Workload Generation](#32-workload-generation)  
   3.3 [Query Execution on Cloudlab Machines](#33-query-execution-on-cloudlab-machines)  
4. [Data Preprocessing](#4-data-preprocessing)  
   4.1 [Parse Query Plans](#41-parse-query-plans)  
   4.2 [Augment Query Plans with Sample Bitmaps](#42-augment-query-plans-with-sample-bitmaps)  
   4.3 [Gather Feature Statistics](#43-gather-feature-statistics)  
5. [Model Training](#5-model-training)  
   5.1 [Local Training](#51-local-training)  
   5.2 [Remote Training](#52-remote-training)
6. [Model Evaluation](#6-model-evaluation)  
   6.1 [Local Inference](#61-local-inference)  
   6.2 [Remote Inference](#62-remote-inference)
7. [Reproducing Results of the Paper](#7-reproducing-results-of-the-paper)  
   7.1 [Inference of Query Optimization Workloads](#71-inference-of-query-optimization-workloads)  
   7.2 [Downloading Pre-collected Model Evaluations](#72-downloading-pre-collected-model-evaluations)  
   7.3 [Generate Paper Plots](#73-generate-paper-plots)  
8. [Source Code Origins](#8-source-code-origins)  
   

   
# 1. Overview
In this repository, we provide the source code to completely reproduce the results of the paper from scratch. 
However, as this is a time-consuming process, we additionally provide all original artifacts (i.e. query plans, execution costs, etc.) 
that we used in the paper. This allows skipping steps such as data collection or model training and allows to directly 
evaluate LCMs on the provided data or just to analyze their predictions.

## 1.1 Overall Workflow
On a high level, fully reproducing the results of the paper requires the following three steps:
1. **Data Collection**: To train and evaluate LCMs, a broad set of query traces (e.g. physical query plans and their
   execution costs/runtime) is required. To this end, our repository provides a set of scripts to collect the data from PostgreSQL databases. 
   In particular, our scripts leverage the scientific research and cloud platform [Cloudlab](https://www.cloudlab.us/) 
   to run the queries on a set of machines, which allows reproducing the results. In addition, we recommend to run 
   queries on such an isolated environment, to avoid interference with other workloads that might bias the runtime.
   To speedup the execution of training and evaliation queries, we used a set of up to 10 Cloudlab c8220 machines.
   We explain the setup in detail in the [Experimental Setup](#3-experimental-setup) section. However, running all 
   queries takes up to several weeks (depending on the parallelism), as the datasets are large and the queries are complex.
   For that reason, we provide a set of **pre-executed training and evaluation queries** that can be 
   downloaded from [OSF](https://osf.io/rb5tn/) as we explain below.
2. **Model Training**: Once the queries are available, the next step is to train the LCMs. 
   For this, we provide a set of scripts that can be used to train the LCMs on the collected data. The training process is 
   described in detail in the [Local Training & Evaluation Pipeline](#4-local-training--evaluation-pipeline) section.
   The training process can be executed on a local machine. However, we recommend to use a set of training machines, as
   the training process is time-consuming and requires a lot of resources. We provide a set of scripts to automate the
   training process on a set of machines in the [Remote Training & Evaluation Pipeline](#5-remote-training--evaluation-pipeline) section.
3. **Model Evaluation**: After the LCMs are trained, they can be evaluated on a set of evaluation queries as described in the paper.
   We provide a set of scripts to evaluate the trained LCMs on the collected data. The evaluation process is described in detail
   in the [Remote Training & Evaluation Pipeline](#5-remote-training--evaluation-pipeline) section.

## 1.2. Local and Remote Operations 
We distinguish between **local** and **remote** operations in the following.
- **Local**: Models are trained and evaluated on the local host, i.e. the machine where the code is executed.
While useful for debugging and development, this is impractical, as datasets are large and the training of the models is time-consuming.
- **Remote**: Models are trained and evaluated on remote machines. This includes the execution of training and evaluation queries, which we conducat on Cloudlab machines.
CLoudlab is a research platform that provides access to a variety of hardware resources for research purposes.
Moreover, we use a set of training machines that are equipped with GPUs to accelerate the training process.

## 1.3 Hardware Instances
This code distinguishes between different type of machines.
- **Localhost**: Can be used to train and evaluate models in local mode. In remote mode, it is used as central coordinator to distribute code and evaluate results.
- **Cloudlab-Instances**: The machines of type [c8220](https://www.clemson.cloudlab.us/portal/show-nodetype.php?type=c8220&_gl=1*1goap78*_ga*MTIyNzE5ODc0My4xNzIxOTIwMTMw*_ga_6W2Y02FJX6*MTcyMzE4MzIwNi42LjEuMTcyMzE4MzIyMy4wLjAuMA..) 
  used in the Cloudlab environment to run training and test queries. These are the target instances, i.e. for these machines and database environments, LCMs are trained.
- **Training-Instances**: In order to accelerate the training LCMs, we use a set of cluster nodes that especially have GPU support (CUDA).
Moreover, we integrated and recommend [Weights & Biases](https://wandb.ai/site) to monitor the training process and evaluate the results.


# 2. Prerequisites
In the following, we describe the prerequisites to run the code locally.

## 2.1. Python environment
To run the code locally, please run the following command to install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements.txt
```

## 2.2 Check environment variables
All file paths and credentials are stored in a `.env` file in the root directory of the repository.
The provided `.env.example` file can be used as a template to create your own `.env` file.
Especially, if you aim for remote execution, please make sure to set the following environment variables in your `.env` file.
In the following, we describe the required environment variables in detail.
```bash
- `LOCAL_ROOT_PATH`: The path to the root directory of the repository on the local host.
- `LOCAL_KNOWN_HOSTS_PATH`: The path to the known_hosts file on the local host.
- `CLOUDLAB_ROOT_PATH`: The path to the root directory of the repository on the cloudlab machines.
- `CLOUDLAB_SSH_USERNAME`: The username to log in to the cloudlab machines.
- `CLOUDLAB_SSH_KEY_PATH`: The path to the ssh key to log in to the cloudlab machines.
- `CLOUDLAB_SSH_PASSPHRASE`: The passphrase of the ssh key to log in to the cloudlab machines.
- `CLUSTER_ROOT_PATH`: The path to the root directory of the repository on the training machines.
- `CLUSTER_STORAGE_PATH`: The path to the storage directory on the training machines.
- `CLUSTER_SSH_USERNAME`: The username to login to the training machines.
- `CLUSTER_SSH_KEY_PATH`: The path to the ssh key to login to the training machines.
- `CLUSTER_SSH_PASSPHRASE`: The passphrase of the ssh key to login to the training machines.
- `WANDB_USER`: The username of the Weights & Biases account.
- `WANDB_PROJECT`: The project name of the Weights & Biases account.
- `OSF_USERNAME`: The username of the OSF account where the data is stored.
- `OSF_PASSWORD`: The password of the OSF account where the data is stored.
- `OSF_PROJECT`: The project name of the OSF account where the data is stored.
- `NODE00`: Node Information of the first training machine in the form of: `{'hostname': 'hostname.com', 'python': '3.9'}`.
```

## 2.3 Download precollected artifacts
To download the pre-collected training and evaluation queries, please run: 
```bash
cd src
python3 download_from_osf.py --artifacts runs 
```
By specifying the `--artifacts` flag, the script downloads corresponding artifacts from the OSF repository.
In particular, we provide the following artifacts:
- `datasets`: These are the tabular CSV datasets that are queried by the training and evaluation queries.
  The datasets are stored in a set of directories that are named after the dataset (e.g. `imdb`, `baseball`).
- `workloads`: These are the SQL queries that are executed against the datasets.
  They are used to generate the physical query plans and their execution costs.
  The SQL queries are stored in a set of directories that are named after the dataset and the workload (e.g. `imdb_scaled1`, `baseball_scaled1`).
  Each directory contains a set of files that are named after the query (e.g. `workload_100k_s1.sql`).
  The file `workload_100k_s1.sql` refers to the training dataset, while other files (e.g. `join_order_full/job_light_36.sql`) refer to the evaluation dataset.
- `runs`: This contains the pre-executed training and evaluation queries, i.e. the physical query plans and their execution costs.  
   The files contain the physical query plans and their execution costs in different formats (e.g. `raw`, `parsed_plan`, `parsed_plan_baseline`, `json`).
   The data is stored in a set of directories that are named after the dataset (e.g. `imdb`, `baseball`).
   Each directory contains a set of files that are named after the query (e.g. `workload_100k_s1.json` for the executed training queries).
   For more details on the formats, please see the OSF repository.
- `models`: This contains the pre-trained LCMs that were saved in pyTorch or pickle format.
   Note that we trained each model three times with different random seeds. 
   Moreover, note that workload-driven models (e.g. `QPPNet`, `End-To-End`, `QueryFormer`, `ZeroShot`, `DACE`) are trained on a target database,
   while the tabular models (e.g. `FlatVector`, `Zero-Shot`) are trained on the other 19 unseen databases.
- `evaluation`: This contains the pre-collected evaluation queries, i.e. the physical query plans and their execution costs.
  Each directory contains a set of files that are named after the workload, benchmark and the query  (e.g. `imdb/join_order_full/job_light_36.json`).
  

# 3.Query Execution
In the following, we describe how to execute training and evaluation queries on the datasets.
To execute queries on your own, machines need to be set up and created. Moreover, workloads needs to be generated.
In the following, we describe how to set up the machines and how to execute the queries.

## 3.1. Downloading query workloads 
However, as mentioned above, we provide pre-executed queries that can be downloaded from the OSF repository.
To download the existing workloads, please run the following command:
```bash
python3 download_from_osf.py --artifacts workloads 
```
## 3.1. Workload Generation
To generate the training and evaluation workloads yourelf, we provide a set of scripts that can be used to generate the SQL queries.
The entrypoint is located at `src/run_benchmark.py`. 
Please first download the CSV datasets as they are required to generate the workloads.:
```bash
python3 download_from_osf.py --artifacts datasets 
```
Now create the workload with the following command. See the `run_benchmark.py` script for more details on the parameters.
```bash
python3 run_benchmark.py --generate_workload --data_dir "$HOME/lcm-eval/datasets/" --target "$HOME/lcm-eval/data/workloads/new_workload.sql" --dataset imdb
```
To additionally create the evaluation workloads that we used in our paper, please run:
```bash
python3 evaluation/workload_creation/create_evaluation_workloads.py
```
## 3.2. Query Execution on Cloudlab Machines
To execute the queries on the Cloudlab machines, we provide a set of scripts that can be used to run the queries.
To reproduce this, please first create an account at Cloudlab and login.
Then, please use the `.rspec` file that is provided in `misc/cloudlab.rspec`, which 
guarantees that the machines are set up correctly and in the same way as we used in the paper.
After the machines are created, you can use the script `parse_cloudlab.py` on the system output to 
automatically extract the node names and SSH configuration for convenience.
To automatically set-up machines, install PostgreSQL, ingest CSV data and execute the workloads, we provide a set of scripts.
Importantly, make sure that you add all hostnames to `misc/hostnames` and that you are able to connect via SSH to the machines.
To run the set-up script, please run the following command.
```bash
scripts/exp_runner/exp_setup.py --task setup deliver start monitor
````
Note that this needs to be executed **twice**, as the hardware requires a reboot in between.
This script installs all necessary dependencies, downloads the required data from the OSF repository,
and ingests the data into PostgreSQL.
After the script terminated, please login to the cloudlab machines and make sure that PostgreSQL is running and the data is ingested.
You can do this by running the following command:
```bash
ssh <username>@<hostname> "psql -U postgres -d imdb -c 'SELECT COUNT(*) FROM movies;'"
```
To finally execute the training and evaluation queries on the remote cloudlab machines, we provide a set of scripts that can be used to run the queries.
Please run the following command to execute the training queries:
```bash
python3 scripts/exp_runner/exp_run_training_workload.py --task setup deliver start monitor
```
Make sure to adapt the script to your needs, e.g. by specifying the workload and the target database, the timeout for the queries, etc.
The script will execute the training queries on the remote machines and store the results in the specified directory.
Once the queries are executed, can pick up the results with the following command:
```bash
python3 scripts/exp_runner/exp_run_training_workload.py --task pickup
```

## 4. Data Preprocessing
To preprocess the data, we provide a set of scripts that can be used to parse the query plans and gather feature statistics.
These preprocessing steps are required to train the LCMs.
We demonstrate the preprocessing steps for the `imdb` dataset and the 200k training workload.
However, the same steps can be applied to other datasets and workloads as well.
To automate these steps, we provide additional scripts at `src/scripts`.
Overall, we importantly support different formats of query plans:
- `raw`: This is the output of Postgres `EXPLAIN ANALYZE`command. Moreover, database and column statistics are stored.
- `parsed_plan`: This is a parsed version of the query plan that is fed into the LCMs.
- `parsed_plan_baseline`: These query plans are also parsed and additionally contain sample bitmaps which are required by some LCM.
- `json`: As identified later, Postgres also directly provides the query plan in JSON format. This format contains more features than the standard plans and is required particularly by `QPPNet`.

## 4.1. Parse Query Plans
To parse the query plans in to a specific format that is required by the LCMs, please run the following command:
```bash
python3 parse_all.py \
  --raw_dir "$HOME/lcm-eval/data/runs/raw" \
  --parsed_plan_dir "$HOME/lcm-eval/data/runs/parsed_plans" \
  --parsed_plan_dir_baseline" $HOME/lcm-eval/data/runs/parsed_plans_baseline" \
  --min_query_ms 0 \
  --max_query_ms 60000000000 \
  --workloads workload_200k_s1 \
  --include_zero_card
  ```
Both outputs are required, the `parsed_plans` is used by the majority of the LCMs, while the `parsed_plans_baseline` is used by the MSCN, that need specific join conditions.

## 4.2. Augment Query Plans with Sample Bitmaps
Some LCMs require sample bitmaps to be able to predict the execution costs of queries (see paper for taxonomy)
To augment the query plans with sample bitmaps, please run the following command:
```bash
python3 baseline.py \
  --augment_sample_vectors \
  --dataset imdb \
  --data_dir "$HOME/lcm-eval/data/datasets/imdb" \
  --source "$HOME/lcm-eval/parsed_plans_baseline/imdb/workload_200k_s1.json" \
  --target "$HOME/lcm-eval/augmented_plans_baseline/imdb/workload_200k_s1.json"
  ````
The resulting converted workloads are stored in the `augmented_plans_baseline` directory.

## 4.3. Gather Feature Statistics
Finally, we need to gather feature statistics from the training workloads to be able to train the LCMs.
They are required to do feature scaling and to provide the LCMs with the necessary information about the data distribution.
To gather the feature statistics, please run the following command:

```bash
python3 gather_feature_statistics.py \
    --database imdb \
    --target "$HOME/lcm-eval/data/runs/feature_statistics/imdb/statistics.json" \
    --workload "$HOME/lcm-eval/data/runs/parsed_plans/imdb/workload_200k_s1.json" \
```
Workload-driven models only need the feature statistics of the corresponding training workload.
In contrast, workload-agnostic models (e.g. `FlatVector`, `Zero-Shot`) require the feature statistics of all training workloads.

# 5. Model Training
Once all the data is at hand and prepared, we can train the LCMs.
Again, they can be trained locally as we will explain below. 
Moreover, we provide a set of scripts to train the LCMs on remote machines automatically.
This server operation is highly recommended, as the training process is time-consuming and requires a lot of resources.
Moreover, we recommend to use a set of training machines that are equipped with GPUs to accelerate the training process.

## 5.1. Local Training
To train a model on your local machine, make sure that you have the required data and feature statistics available.
Then, you can run the following command to train a model:
```bash
python main.py \
  --mode train \
  --wandb_project lcm \
  --wandb_name my_model_training_run \
  --model_type flat \
  --device cuda:0 \
  --model_dir "$HOME/lcm-eval/models/flat_vector/" \
  --target_dir "$HOME/lcm-eval/evaluation/flat_vector/" \
  --statistics_file "$HOME/lcm-eval/data/runs/augmented_plans_baseline/imdb/statistics.json"/ \
  --seed 3 \
  --workload_runs  "$HOME/lcm-eval/data/runs/parsed_plans/imdb/workload_100k_s1_c8220.json" \
  --test_workload_runs "$HOME/lcm-eval/data/runs/parsed_plans/imdb/join_order_full/job_light_36.json"
 ```

Here, each flag is explained in detail:
- `mode`: The mode of the script, i.e. `train`, `retrain` or `predict`.
- `wandb_project`: The name of the Weights & Biases project to log the training process.
- `wandb_name`: The name of the Weights & Biases run to log the training process.
- `model_type`: The type of the model to train, i.e. `flat`, `mscn`, `qppnet`, `e2e`, `queryformer`, `zeroshot`, `dace`.
- `device`: The device to use for training, i.e. `cuda:0` for GPU or `cpu` for CPU.
- `model_dir`: The directory where the trained model is saved.
- `target_dir`: The directory where the predictions are saved.
- `statistics_file`: The path to the feature statistics file that is used to train the model.
- `seed`: The random seed to use for training.
- `workload_runs`: The path to the training workload runs that are used to train the model.
- `test_workload_runs`: The path to the test workload runs that are used to evaluate the model.

## 5.2. Remote Training
To train the LCMs on remote machines, we provide a set of scripts that can be used to automate the training process.
The entrypoint is located at `src/scripts/exp_runner/exp_train_model.py`.
To train the LCMs on the remote machines, please run the following command:
```bash
python3 exp_train_model.py --task setup deliver start monitor
```
This command will set up the remote machines, deliver the code and data, start the training process, and monitor the training process using wandb.
Make sure to set the correct model architecture, the target database and the training workload in the script.

# 6. Model Evaluation
We support two modes of evaluation. At first, LCMs are directly tested against unseen test workloads after the training ended.
Please specify `test_workload_runs` for this. Secondly, we enable inference after the LCMs are trained with the `predict` mode.

## 6.1. Local Inference
To evaluate the LCMs on the test workloads, please run the following command:
```bash
python main.py \
    --mode predict \
    --wandb_project lcm \
    --wandb_name my_model_evaluation_run \ 
    --model_type flat \
    --device cuda:0 \
    --model_dir "$HOME/lcm-eval/models/flat_vector/" \
    --target_dir "$HOME/lcm-eval/evaluation/flat_vector/" \
    --statistics_file "$HOME/lcm-eval/data/runs/augmented_plans_baseline/imdb/statistics.json" \
    --seed 3 \
    --test_workload_runs "$HOME/lcm-eval/data/runs/parsed_plans/imdb/join_order_full/job_light_21.json"
```
This command will evaluate the trained model on a given test workload and log the results to Weights & Biases.

## 6.2. Remote Inference
To evaluate the LCMs on the remote machines, we provide a set of scripts that can be used to automate the evaluation process.
The entrypoint is located at `src/scripts/exp_runner/exp_predict_all.py`.
To evaluate the LCMs on the remote machines, please run the following command:
```bash
python3 exp_predict_all.py --task setup deliver start monitor
```
This command will set up the remote machines, deliver the code and data, start the evaluation process
and monitor the evaluation process using Weights & Biases.
Make sure to set the correct model architecture, the target database and the test workload in the script.

# 7. Reproducing Results of the Paper
Fully reproducing the results of the paper requires running the training and evaluation queries on the remote machines.
Moreover, the LCMs need to be trained on the collected data and evaluated on the evaluation queries.

## 7.1. Inference of Query Optimization Workloads
In particular, make sure to run the inference on the following evaluation workloads, that we used in the paper:
- **Join Order Experiment**: This experiment was executed on JOB, which operates on IMDB. To this end, the workload files are named according to the scheme:
  `imdb/join_order_full/*.json`, where each file contains all enumerated query plans for one query of the JOB benchmark.
- **Access Path Selection**: This exeriment was executed on IMDB, Baseball and TPC-H. Here, we iterated through the selectivity
  space in 10% steps. Thus, you can find the files according to the scheme  `<DATABASE>/scan_costs_percentiles/<IDX/SEQ><table>.<column>/*.json`.
- **Physical Operator Selection**: This experiment was executed on IMDB, Baseball and TPC-H. Here, we iterated through the physical operators
  in the query plans. Thus, you can find the files according to the scheme `<DATABASE>/physical_plan/*.json`.
  Here, each workload file contains the same query three times, each with a different physical operator.
- **Physical Operator Selection with additional Indexes**: This experiment was executed on IMDB. We additionally built indexes on all tables and all columns of IMDB and executed the same queries as in the previous experiment.
  Thus, you can find the files according to the scheme `<DATABASE>/physical_index_plan/*.json`.
- **Retraining Workloads**: We additionally provide the workloads and query traces for the retrainng experiment. Here, we executed table scans with both Sequential and Index scan and fine-tuned LCMs with this.
  The corresponding files are stored for IMDB under `<DATABASE>/index_retraining/*.json` and `<DATABASE>/seq_retraining/*.json`.

## 7.2. Downloading Pre-collected Model Evaluations
In our data repository, we provide the model evaluations that can be downloaded from the OSF repository.
These include the predictions for the test workloads and importantly for the query optimization evaluation workloads.
Each model was trained three times with different random seeds, which is why we provide the predictions in a set of directories that are named after the model and the seed.
To download the pre-collected model evaluations, please run the following command:
```bash
python3 download_from_osf.py --artifacts evaluation
```

## 7.3. Generate Paper Plots
To generate the plots for the paper, we provide a set of jupyter notebooks that can be used to evaluate the results of the LCMs and compare them to the ground truth.
The notebooks are located in the `src/evaluation/plots` directory.
This directory contains the following notebooks:
- `01_join_order.ipynb`: This notebook contains the evaluation of the join order experiment.
- `02_join_order_pg_act_card.ipynb`: Here we evaluate the join order experiment with the PostgreSQL cardinality estimates.
- `03_access_path_selection.ipynb`: This notebook contains the evaluation of the access path selection experiment.
- `04_physical_plan_selection.ipynb`: This notebook contains the evaluation of the physical operator selection experiment.

# 8. Source Code Origins
We thank all authors of the following repositories for providing their code. 
We modified some parts of the code to fit our evaluation setup, but the core functionality remains the same.

- **FlatVector** Predicting Multiple Metrics for Queries: Better Decisions Enabled by Machine Learning (ICDE 2009)
  -Authors: Archana Ganapathi; Harumi Kuno; Umeshwar Dayal; Janet L. Wiener; Armando Fox; Michael Jordan
  - Paper: https://ieeexplore.ieee.org/document/4812438
  - Code: Not available, but easy to implement (we used the simple flat vector model of this approach) 
    Each query is expressed as a feature vector that contains out of the operator types and their summed, intermediate cardinalities
  - Modifications: We applied a state-of-the-art regression on top (LightGBM) to predict the cost of the query. 
    The code is available at `src/models/tabular/train_tabular_baseline.py`.


- **MSCN**: Learned Cardinalities: Estimating Correlated Joins with Deep Learning (CIDR 2019)
  - Authors: Andreas Kipf, Thomas Kipf, Bernhard Radke, Viktor Leis, Peter Boncz, Alfons Kemper
  - Paper: https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf
  - Code: https://github.com/andreaskipf/learnedcardinalities
  - Modifications: While originally designed for cost estimation, we use MSCN to predict execution costs (i.e. runtime) of queries.
    Most importantly, MSCN only learns from SQL strings and for sample bitmaps.
    Thus, we augment all plans with sample bitmaps, which are in the corresponding `augmented_plan_baseline` files.
    Moreover, as MSCN relies on one-hot-Encoding, it is built dynamically fo a given dataset based on the extracted feature-statistics.


- **QPPNet**: Plan-Structured Deep Neural Network Models for Query Performance Prediction (VLDB 2019)
  - Authors: Ryan Marcus, Olga Papaemmanouil
  - Paper: https://dl.acm.org/doi/abs/10.14778/3342263.3342646
  - Code: https://github.com/rabbit721/QPPNet
  - Modifications: While the original QPPNet-Implementation was tailored towards three datasets (TPC-H), we adapted the code to work with our datasets.
    We dynamically read out feature statistics to directly use the correct model size according to the required encoding.
    In addition, we executed training and evaluation queries in `JSON` mode for this particular model (i.e. `EXPLAIN ANALYZE FORMAT JSON`), 
    as it requires more features that are not available in the standard query plans (i.e. `EXPLAIN ANALYZE`).
    The code is available at `src/models/qppnet/`.
  

- **End-To-End**: End-To-End Learning for Cost Estimation of Query Execution Plans (VLDB 2019)
  - Authors: Ji Sun, Guoliang Li
  - Paper: https://dl.acm.org/doi/abs/10.14778/3368289.3368296
  - Code: https://github.com/greatji/Learning-based-cost-estimator
  - Modifications: Similar to MSCN, this model requires sample bitmaps. We augmented the query plans with sample bitmaps and trained the model.
    We also need to adapt the source code to be applicable for different datasets.
    The main implementation is available at: `src/models/workload_driven/model/mscn_model.py
  

- **QueryFormer**: QueryFormer: A Tree Transformer Model for Query Plan Representation (VLDB 2022)
  - Authors: Yue Zhao, Gao Cong, Jiachen Shi, Chunyan Miao
  - Paper: https://www.vldb.org/pvldb/vol15/p1658-zhao.pdf
  - Code: https://github.com/zhaoyue-ntu/QueryFormer
  - Modifications: Query Former was hard-coded to the IMDB dataset and often used non-parametric variables. T
    Thus, we deciced to extract the PyTorch model and ingested it in our existing training pipeline.
    We also adapted the code to work with our datasets and to be more flexible in terms of the input data.
    The implementation is available at: `src/models/workload_driven/model/e2e_model.py`
  
- **ZeroShot**: Zero-shot Cost Estimation with Deep Learning (VLDB 2022)
  - Authors: Benjamin Hilprecht and Carsten Binnig
  - Paper: https://www.vldb.org/pvldb/vol15/p2361-hilprecht.pdf
  - Code: https://github.com/DataManagementLab/zero-shot-cost-estimation
  - Modifications: Little modifications were applied to this model, as its source code was used and extended as basis for our paper.
    However, various improvements were made to the code to make it more efficient and to adapt it to our datasets.
    Moreover, we implemented different model variants and an optimized training pipeline.
    The implementation is available at: `src/models/zeroshot/zero_shot_model.py` 


- **DACE**: DACE: A Database-Agnostic Cost Estimator (ICDE 2024)
  - Authors: Zibo Liang, Xu Chen, Yuyang Xia, Runfan Ye, Haitian Chen, Jiandong Xie, Kai Zheng 
  - Paper: https://ieeexplore.ieee.org/document/10598079
  - Code: https://github.com/liang-zibo/DACE
  - Modifications: As DACE builds up on the Zero-Shot code base, its adaption was straight forward and little changes were required to make it running.
    Like in the other models, we adapted the code to work with our datasets and to be more flexible in terms of the input data.
    The impelementation is available at: `src/models/dace/dace_model.py`
