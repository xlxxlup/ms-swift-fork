#!/bin/bash

# 使用Megatron-DeepSpeed进行LoRA微调Qwen3-30B-A3B模型
# 数据路径
# 多卡LoRA微调Qwen3-30B-A3B模型
# 数据路径
DATA_PATH="/data2/summary_data_replaced_with_line_1k.jsonl"

# 输出路径
OUTPUT_DIR="/data2/output/qwen3_30b_a3b_lora_pifu_dialogue_0309"

# 使用的GPU数量
NUM_GPUS=4

# 设置CUDA内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Megatron并行配置
# 对于4卡，使用 tp=2, pp=2 的组合
TP=4
PP=4







# 启动Megatron训练
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29503 \
python -m swift.megatron.cli.sft \
    --model_type qwen3 \
    --template qwen3 \
    --model /data2/Qwen3-30B-A3B-Instruct-2507 \
    --dataset $DATA_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --eval_strategy steps \
    --optim adamw_torch \
    --logging_steps 2 \
    --save_steps 100 \
    --save_total_limit 4 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --gradient_checkpointing true \
    --lora_rank 32 \
    --lora_alpha 64 \
    --loss_scale last_round \
    --lora_dropout 0.05 \
    --max_length 2048 \
    --load_from_cache_file false \
    --output_dir $OUTPUT_DIR \

    --ddp_find_unused_parameters false \
    --save_only_model true \
    --report_to swanlab \
    --swanlab_project qwen3-dialogue-model \
    --swanlab_exp_name qwen3-30b-a3b-megatron-lora_0302 \
    --swanlab_mode cloud \
    --enable_thinking false \
    --tensor_parallel_size $TP \
    --pipeline_parallel_size $PP \
    --sequence_parallel true
