#### Introduction
This is the official implementation (code & dataset) to our paper: [Could Small Language Models Serve as Recommenders? Towards Data-centric Cold-start Recommendations](https://arxiv.org/abs/2306.17256). 
This repo includes the first benchmark for the system cold-start recommendation problem and provides a prompt learning based baseline for this task. We also propose two data-centric strategies to enhance _SMALL_ language models for in-context recommendations.

#### Benchmark

We download the three datasets from their official websites and keep the same file at ``datasets/downstream_tasks`` for your convenience.  All datafiles have the same MD5 codes as the files you download from official websites by yourself.

The data preparing code is located at `src/prepare_datasets.py`, you may run it with a specific random seed as: ``python src/prepare_datasets.py 42``. The following experiments will automatically call this code over five different random seeds. 

_Warning:_ If you want to run multiple random seeds simulatively, you need to _COPY and PASTE_ the _entire_ project to another path. Current ``prepare_datasets.py`` is implemented _IN PLACE_!

#### Implementation 

* PromptRec is implemented at ``src/models/ours.py`` with the class name: ``TransferSoftPromptRecommander``, it also supports the transferable prompt pre-training strategy.
* Templates and verbalizers are defined at ``src/datautils/coupons.py``, ``src/datautils/movielens.py``, ``src/datautils/restaurants.py``. 
* ``src/auto_report.py``: automatically extract the results in log files and compute the Mean and Std over multiple random seeds.
* ``src/run_rcmp.sh``, ``src/run_tppt.sh``, and ``src/run_promptrec.sh``: run the corresponding experiments over five random seeds.

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

* Experiments 2: PromptRec (Table 2)

```shell
>>> nohup sh run_promptrec.sh 0 > logs/zeroshot.log &
>>> python auto_report.py logs/zeroshot.log
>>> cat logs/zeroshot_mean.tsv logs/zeroshot_std.tsv
```

* Experiments 3: Transferable prompt pre-training (Table 4)

```shell
# It could be several days to run it with single A6000 GPU.
>>> nohup sh run_tppt.sh 0 > logs/tppt.log &
>>> python auto_report.py logs/tppt.log
>>> cat logs/tppt_mean.tsv logs/tppt_std.tsv
```

* Experiments 4: Refine corpurs for model pre-training (Table 4)

```shell
# Equation-8 is implemented in corpus_selection.py
# the following script will split the workload on 4 RTX-3070 GPU, but it still needs at least 6 weeks to prepare the extracted corpus for each domain. you may modify this strategy according to your computing resources. 
# note that, the output file could be very large, so you need to worry about storing resources as well.
>>> nohup sh run_corpus_refine.sh & 

# you may need to merge four files together MANUALLY, they are located at ../datasets/refined_corpus folder. for example, the code on Linux could be:  
>>> cat split_1.txt > full.txt && cat split_2.txt >> full.txt ....

# assuming that your full parsed corpus is located in the file c4_$dataset$_skip20.txt
# then you can use corpus_selection.py again to sampling a subset with top-K documents.
# In the following, we only keep the top-10000 documents for each domain. 
# The output file will looks like: top10000_c4_$dataset$_skip20.txt at the same folder.
>>> python corpus_selection.py ../datasets/refined_corpus/c4_coupon_skip20.txt 10000
>>> python corpus_selection.py ../datasets/refined_corpus/c4_ml100k_skip20.txt 10000
>>> python corpus_selection.py ../datasets/refined_corpus/c4_restaurant_skip20.txt 10000

# now, we will further pre-train our model on our refined corpus, and then test its performance with PromptRec.
# the shell run_rcmp.sh will first randomly shuffle the datasets, and further pre-train a language model, finally verify the pre-trained model.
# You could find the pre-training scripts from the folder src/pretrain_scripts/, and copy it to the src/ folder. 
# around 10 hours for each command on a single A6000 GPU
>>> nohup sh run_rcmp.sh 0 ../datasets/refined_corpus/top10000_c4_coupon_skip20.txt coupon 10000 > logs/rcmp_coupon.log &  

>>> python auto_report.py logs/rcmp_coupon.log 
>>> cat logs/rcmp_coupon_mean.tsv logs/rcmp_coupon_std.tsv
```

* Tips: You may download the .tsv files and open them with Excel for better visualization.
