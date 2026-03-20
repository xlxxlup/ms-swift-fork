#!/bin/bash

# 多卡Megatron-LM微调Qwen3-30B-A3B模型
# 数据路径
DATA_PATH="/data2/summary_data_replaced_with_line_1k.jsonl"

# 输出路径
OUTPUT_DIR="/data2/output/qwen3_30b_a3b_megatron_pifu_dialogue"

# 启动Megatron训练
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model /data2/Qwen3-30B-A3B-Instruct-2507 \
    --dataset $DATA_PATH \
    --save_safetensors true \
    --load_from_cache_file false \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --packing false \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --logging_steps 2 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir $OUTPUT_DIR \
    --save_steps 100 \
    --save_total_limit 4 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --report_to swanlab \
    --swanlab_project qwen3-dialogue-model \
    --swanlab_exp_name qwen3-30b-a3b-megatron-dialogue
