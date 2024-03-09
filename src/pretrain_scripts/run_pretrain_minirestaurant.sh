export CUDA_VISIBLE_DEVICES=$2

python -u run_mlm.py \
       --output_dir outputs/$5/10k_$4_seed$1 \
       --save_steps 500 \
       --learning_rate 4e-5 \
       --model_name_or_path prajjwal1/bert-$4 \
       --train_file $3  \
       --adam_beta2 0.95 \
       --adam_beta1 0.9 \
       --adam_epsilon 1e-8 \
       --seed $1  \
       --per_device_train_batch_size 32  \
       --do_train \
       --do_eval \
       --max_seq_length 256 \
       --fp16 true \
       --max_steps 10000 \
       --warmup_ratio 0.05 \
       --weight_decay 0.0 \
       --label_smoothing_factor 0. \
       --gradient_accumulation_steps 1 \
       --adafactor true \
       --lr_scheduler_type linear 
