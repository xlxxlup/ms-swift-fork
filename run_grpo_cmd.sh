#!/bin/bash
#
# GRPO 训练启动脚本 (使用 swift rlhf 命令) - 8 GPU
#
# 使用方法:
#   chmod +x run_grpo_cmd.sh
#   ./run_grpo_cmd.sh
#

# 设置可见的 GPU (8张卡)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 训练参数
MODEL_PATH="/data2/Qwen3-14B-Instruct"
SFT_ADAPTER="/data2/ms-swift/ms-swift/output/qwen3-reward-model-sft-lora-0313/v1-20260313-133450/checkpoint-250"
DATA_DIR="/data2/皮肤科/强化学习数据"
OUTPUT_DIR="${DATA_DIR}/grpo_output"

# 自定义奖励函数权重
# correctness: 正确性奖励 (chosen > rejected → +2分)
# rule_match: 规则匹配奖励 (每匹配一个规则 → +1分)
# order_consistency: 顺序一致性奖励 (每组2条数据得分关系一致 → 各+2分)

# 使用 torchrun 启动多 GPU 分布式训练
NPROC_PER_NODE=8 \
torchrun --nproc_per_node=8 \
    /data2/ms-swift/ms-swift/swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --model_type qwen3 \
    --template qwen3 \
    --adapters "${SFT_ADAPTER}" \
    --reward_funcs correctness rule_match \
    --reward_weights 2.0 1.0 \
    --dataset "${DATA_DIR}/grpo/grpo_dataset.jsonl" \
    --dataset_shuffle false \
    --load_from_cache_file true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --tuner_type lora \
    --tuner_backend peft \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --num_generations 4 \
    --num_iterations 1 \
    --max_completion_length 8192 \
    --temperature 0.9 \
    --top_p 0.9 \
    --top_k 50 \
    --stop_words "</score>" \
    --repetition_penalty 1.1 \
    --beta 0.01 \
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
    --swanlab_exp_name qwen3-14B-Intruct-grpo-0317 \
    --log_completions true \
    --log_entropy true \
    --vllm_tensor_parallel_size 2  # 每个模型放几张卡上
