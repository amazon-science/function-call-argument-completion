# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# key arguments:
# --do_eval_fc=$CALLDIR >> evaluate on the callargs
# --usages >> maximum number of usages
# --use_implementation
# --model_type=codegen >> don't forget to specify model_type
# Model training for Table 4 results (RQ 1 & RQ 2.1)

LANG=python                      
DATADIR=/home/ubuntu/callargs_general_completion/ # can be found in s3://pyenvs-and-callargs/callargs_general_completion
LITFILE=/home/ubuntu/code/CodeXGLUE/Code-Code/CodeCompletion-token/dataset/py150/literals.json
OUTPUTDIR=/home/ubuntu/results/codegen_dist_completion/
PRETRAINDIR=Salesforce/codegen-350M-mono
LOGFILE=codegen_dist_completion.log
PER_NODE_GPU=2       # modify YOUR_GPU_NUM

CUDA_VISIBLE_DEVICES=0,1 python run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=codegen \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=5e-5 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=2 \
        --gradient_accumulation_steps=8 \
        --num_train_epochs=10 \
        --logging_steps=100 \
        --weight_decay 0.01 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain 2>&1 | tee tee_codegen_dist_completion.log

# Evaluation

LANG=python                     
DATADIR=/home/ubuntu/callargs_general_completion/
LITFILE=/home/ubuntu/code/CodeXGLUE/Code-Code/CodeCompletion-token/dataset/py150/literals.json
CALLDIR=/home/ubuntu/callargs/pkl_data/ # CALLARGS dataset
OUTPUTDIR=/home/ubuntu/results/codegen_dist_completion_6000_eval
PRETRAINDIR=/home/ubuntu/results/codegen_dist_completion/checkpoint-6000-2.4666 # select a suitable checkpoint
LOGFILE=codegen_dist_completion_6000_eval.log

# token-level accuracy 

CUDA_VISIBLE_DEVICES=0 python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=codegen \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=1000 \
        --seed=42 2>&1 | tee codegen_6000.log

# local context
CUDA_VISIBLE_DEVICES=0 python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=codegen \
        --block_size=1024 \
        --do_eval \
        --do_eval_fc=$CALLDIR \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=1000 \
        --seed=42 2>&1 | tee codegen_6000_ori.log
        
LANG=python                     
DATADIR=/home/ubuntu/callargs_general_completion/
LITFILE=/home/ubuntu/code/CodeXGLUE/Code-Code/CodeCompletion-token/dataset/py150/literals.json
CALLDIR=/home/ubuntu/callargs/pkl_data/
OUTPUTDIR=/home/ubuntu/results/codegen_dist_completion_6000_imp
PRETRAINDIR=/home/ubuntu/results/codegen_dist_completion/checkpoint-6000-2.4666     
LOGFILE=codegen_dist_completion_6000_imp.log

# use implememtation
CUDA_VISIBLE_DEVICES=0 python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=codegen \
        --block_size=1024 \
        --do_eval \
        --use_implementation \
        --do_eval_fc=$CALLDIR \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=1000 \
        --seed=42 2>&1 | tee codegen_6000_imp.log
        
LANG=python                     
DATADIR=/home/ubuntu/callargs_general_completion/
LITFILE=/home/ubuntu/code/CodeXGLUE/Code-Code/CodeCompletion-token/dataset/py150/literals.json
CALLDIR=/home/ubuntu/callargs/pkl_data/
OUTPUTDIR=/home/ubuntu/results/codegen_dist_completion_6000_use
PRETRAINDIR=/home/ubuntu/results/codegen_dist_completion/checkpoint-6000-2.4666     
LOGFILE=codegen_dist_completion_6000_use.log

# use usages
CUDA_VISIBLE_DEVICES=0 python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=codegen \
        --block_size=1024 \
        --do_eval \
        --usages=3 \
        --do_eval_fc=$CALLDIR \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=1000 \
        --seed=42 2>&1 | tee codegen_6000_use.log