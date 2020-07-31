#!/bin/sh
python run_language_modeling.py \
    --output_dir='model_20200731_121800' \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --save_total_limit=5 \
    --num_train_epochs=5.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=500 \
    --save_steps=500 \
    --train_data_file=data/Ethics_train.txt \
    --do_eval \
    --eval_data_file=data/Ethics_test.txt \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=2 \
    --block_size=128 \
    --gradient_accumulation_steps=5