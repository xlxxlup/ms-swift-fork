#!/bin/bash
#
# 全参微调训练启动脚本 - DeepSpeed Zero-2 + 梯度检查点
#
# 特性:
# - 全参数训练 (Full Fine-tuning)
# - DeepSpeed ZeRO-2 优化
# - 梯度检查点 (节省显存)
#
# 使用方法:
#   chmod +x run_full_train.sh
#   ./run_full_train.sh
#

# 设置可见的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 训练参数
MODEL_PATH="/data2/皮肤科/强化学习数据/qwen3-14b-with-sft"
DATA_DIR="/data2/皮肤科/强化学习数据"
OUTPUT_DIR="${DATA_DIR}/full_grpo_train_output"

# 使用 torchrun 启动多 GPU 分布式训练
NPROC_PER_NODE=8 \
torchrun --nproc_per_node=8 \
    /data2/ms-swift/ms-swift/swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --model_type qwen3 \
    --template qwen3 \
    --reward_funcs correctness rule_match \
    --reward_weights 2.0 1.0 \
    --dataset "${DATA_DIR}/grpo/grpo_dataset.jsonl" \
    --dataset_shuffle false \
    --load_from_cache_file true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.1 \
    --tuner_type full \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --num_generations 4 \
    --num_iterations 2 \
    --max_completion_length 4096 \
    --temperature 0.9 \
    --top_p 0.9 \
    --top_k 50 \
    --stop_words "</score>" \
    --repetition_penalty 1.1 \
    --beta 0.04 \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 4 \
    --overwrite_output_dir true \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --bf16 true \
    --remove_unused_columns false \
    --truncation_strategy left \
    --report_to swanlab \
    --swanlab_project qwen3-14B-Intruct-grpo \
    --swanlab_exp_name qwen3-14B-grpo-full-train-zero2 \
    --log_completions true \
    --log_entropy true \
    --vllm_tensor_parallel_size 8
