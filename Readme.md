## Introduction
This is the official impelementation (code & dataset) to our paper: [Towards Personalized Cold-Start Recommendation with Prompts](https://arxiv.org/abs/2306.17256). 
This repo includes the first benchmark for the system cold-start recommendation problem and provides a prompt learning based baseline for this task.

#### Benchmark

We download the three datasets from their official websites and keep the same file at ``datasets/downstream_tasks`` for your convenience.  All datafiles have the same MD5 codes as the files you download from official websites by yourself.

The data preparing code is located at `src/prepare_datasets.py`, you may run it with a specific random seed as: ``python src/prepare_datasets.py 42``. The following experiments will automatically call this code over five different random seeds. 

#### Implementation 

* PromptRec is implemented at ``src/models/ours.py`` with the class name: ``PromptRecommander``.
* Templates and verbalizers are defined at ``src/datautils/coupons.py``, ``src/datautils/movielens.py``, ``src/datautils/restaurants.py``. 

#### Reproduction

* Setup: Assuming you manage the environment with Conda library.

```shell
>>> cd src
>>> conda create -n PromptLM4CSR python=3.9
>>> conda activate PromptLM4CSR
>>> sh setup.sh
```

* Experiments 1: Baselines (Table 2).

```shell
>>> nohup sh run_baselines.sh 0 > logs/baselines.log &
>>> python auto_report.py logs/baselines.log
>>> cat logs/baselines_mean.tsv logs/baselines_std.tsv
```

* Experiments 2: Cold-start PromptRec (Table 2 and Table 3)

```shell
>>> nohup sh run_zeroshot.sh 0 > logs/zeroshot.log &
>>> python auto_report.py logs/zeroshot.log
>>> cat logs/zeroshot_mean.tsv logs/zeroshot_std.tsv
```

* Tips: You may download the .tsv files and open them with Excel for better visualization.
