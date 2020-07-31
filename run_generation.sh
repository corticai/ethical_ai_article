#!/bin/sh
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=model \
    --k=50 \
    --num_return_sequences=10 \
    --length=512
