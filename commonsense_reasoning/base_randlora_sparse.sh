# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

if [ "$5" = "Qwen2" ]
then
    base='Qwen/Qwen2-0.5B-Instruct'
    model='Qwen2'
    bs=16
fi
if [ "$5" = "LLama3" ]
then
    base='meta-llama/Meta-Llama-3-8B'
    model='LLaMA3-8B'
    bs=16
fi
if [ "$5" = "LLama3-70" ]
then
    base='meta-llama/Meta-Llama-3-70B'
    model='LLaMA3-70B'
    bs=16
fi
if [ "$5" = "Phi3" ]
then
    base='microsoft/Phi-3-mini-4k-instruct'
    model='Phi3'
    bs=16
fi
echo $base

CUDA_VISIBLE_DEVICES=$4 python finetune.py \
    --base_model $base \
    --data_path commonsense_$6k.json \
    --output_dir $3 \
    --batch_size 16  --micro_batch_size $bs --num_epochs 1 \
    --learning_rate 1e-4 --cutoff_len 256 \
    --adapter_name randlora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]'\
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing --sparse

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
		    --model $model \
		    --adapter RandLora \
		    --dataset openbookqa \
		    --base_model $base \
		    --batch_size 1 \
		    --lora_weights $3|tee -a $3/openbookqa.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
		    --model $model \
		    --adapter RandLora \
		    --dataset ARC-Challenge \
		    --base_model $base \
		    --batch_size 1 \
		    --lora_weights $3|tee -a $3/ARC-Challenge.txt


CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model $model \
    --adapter RandLora \
    --dataset boolq \
    --base_model $base \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/boolq.txt


CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model $model \
    --adapter RandLora \
    --dataset social_i_qa \
    --base_model $base \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model $model \
    --adapter RandLora \
    --dataset piqa \
    --base_model $base \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/piqa.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model $model \
    --adapter RandLora \
    --dataset hellaswag \
    --base_model $base \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/hellaswag.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model $model \
    --adapter RandLora \
    --dataset winogrande \
    --base_model $base \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/winogrande.txt

CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
    --model $model \
    --adapter RandLora \
    --dataset ARC-Easy \
    --base_model $base \
    --batch_size 1 \
    --lora_weights $3|tee -a $3/ARC-Easy.txt



