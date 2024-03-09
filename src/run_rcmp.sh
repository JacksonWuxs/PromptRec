# nohup run_rcmp.sh GPUID Corpus_path subset size 
python prepare_datasets.py 41905
sh run_pretrain_$4$3.sh 41905 $1 $2 $4 $3  
python -u RCMP.py 41905 $1 $3 $4 
python prepare_datasets.py 1640 
sh run_pretrain_$4$3.sh 1640 $1 $2 $4 $3  
python -u RCMP.py 1640 $1 $3 $4
python prepare_datasets.py 18025
sh run_pretrain_$4$3.sh 18025 $1 $2 $4 $3  
python -u RCMP.py 18025 $1 $3 $4
python prepare_datasets.py 14629
sh run_pretrain_$4$3.sh 14629 $1 $2 $4 $3  
python -u RCMP.py 14629 $1 $3 $4
python prepare_datasets.py 48265
sh run_pretrain_$4$3.sh 48265 $1 $2 $4 $3  
python -u RCMP.py 48265 $1 $3 $4 
