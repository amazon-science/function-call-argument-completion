# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# key arguments:
# --max_signature_length >> length budget for implementation, 0 for not using implementation
# --usages >> whether to use usages
# --max_references_length >> total length budget for 
# --max_references 3 >> maximum number of usages
# --decoder-only >> to finetune a decoder-only model
# --max_source_left_length >> for unidirectional prediction, usually 0.75*max_source_length
# --model_type >> don't forget to specify model_type

# --data_dir should be set as the local path of s3://pyenvs-and-callargs/callargs/pkl_data.zip (unzip it)
# A note for the future re-runing experiments: --max_target_length can be much smaller

# Example 1
# using function implementation (input length 512), codet5-small for unidirectional prediction
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_gen.py  \
  --do_train --do_eval --do_test --task single_left --data_num -1 --use_implementation \
  --num_train_epochs 10 --warmup_steps 4000 --learning_rate 2e-5 --patience 3 \
  --tokenizer_name=Salesforce/codet5-small  --model_name_or_path=Salesforce/codet5-small --data_dir /home/ubuntu/callargs/pkl_data/  \
  --no_cache --output_dir /home/ubuntu/results/t5_left4_dist_imp \
  --save_last_checkpoints --always_save_model \
  --train_batch_size 64 --eval_batch_size 64 --max_source_length 512 --max_target_length 100 --max_signature_length 64 2>&1 | tee t5_left4_dist_imp.log

# Example 2
# using all contexts (input length 1024), codet5-base for in-filling prediction
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_gen.py  \
  --do_train --do_eval --do_test --task single --data_num -1 --use_implementation --usages \
  --num_train_epochs 10 --warmup_steps 4000 --learning_rate 2e-5 --patience 3 \
  --tokenizer_name=Salesforce/codet5-base  --model_name_or_path=Salesforce/codet5-base --data_dir /home/ubuntu/callargs/pkl_data/  \
  --no_cache --output_dir /home/ubuntu/results/t5_base_full4_dist_refer \
  --save_last_checkpoints --always_save_model \
  --train_batch_size 32 --gradient_accumulation_steps 2 --eval_batch_size 32 --max_source_length 1024 --max_source_left_length 768 --max_target_length 100 \
  --max_signature_length 128 --max_references_length 384 --max_references 3 2>&1 | tee t5_base_full4_dist_refer.log

# Example 3
# Finetune unixcoder (decoder-only) using local context only
# (if we want to run the encoder-decoder version of unixcoder, just remove --decoder_only)
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_gen.py  \
  --do_train --do_eval --do_test --task single_left --decoder_only --data_num -1 --model_type roberta \
  --num_train_epochs 5 --warmup_steps 4000 --learning_rate 2e-5 --patience 3 \
  --tokenizer_name=microsoft/unixcoder-base  --model_name_or_path=microsoft/unixcoder-base --data_dir /home/ubuntu/callargs/pkl_data/ \
  --no_cache --output_dir /home/ubuntu/results/unix_dist \
  --save_last_checkpoints --always_save_model \
  --train_batch_size 32 --gradient_accumulation_steps 2 --eval_batch_size 32 --max_source_length 512 --max_target_length 100 2>&1 | tee unix_dist.log

# Example 4
# evaluate CDI setting on a existing model
# here need to manually ensure --output_dir has the trained model weights (copy the intended directory)
CUDA_VISIBLE_DEVICES=0 python run_gen.py  \
  --do_test --task single --data_num -1 \
  --num_train_epochs 10 --warmup_steps 4000 --learning_rate 2e-5 --patience 3 --use_implementation --usages \
  --tokenizer_name=Salesforce/codet5-base  --model_name_or_path=Salesforce/codet5-base --data_dir /home/ubuntu/callargs/pkl_data/  \
  --no_cache --output_dir /home/ubuntu/results/t5_base_full4_dist_imp_eval_ref \
  --save_last_checkpoints --always_save_model \
  --train_batch_size 32 --gradient_accumulation_steps 2 --eval_batch_size 32 --max_source_length 512 --max_source_left_length 384 --max_target_length 100 \
  --max_signature_length 64 --max_references_length 192 --max_references 3 2>&1 | tee t5_base_full4_dist_imp_eval_ref.log