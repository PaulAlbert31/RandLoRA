rank=20
name=randlora${rank}
iter=26290

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --eval_interval 2000 \
    --save_interval 1000 \
    --lora_dim $rank \
    --lora_alpha $((2*rank)) \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/${name}_e2e \
    --fp16 \
    --random_seed 1 \
    --randlora

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/${name}_e2e/model.${iter}.pt \
    --platform local \
    --lora_dim ${rank} \
    --lora_alpha $((rank*2)) \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/${name}_e2e \
    --output_file predict.${iter}.b10p08r4.jsonl \
    --random_seed 1 \
    --randlora

python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/${name}_e2e/predict.${iter}.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref_${name}.txt \
    --output_pred_file e2e_pred_${name}.txt

python eval/e2e/measure_scores.py e2e_ref_${name}.txt e2e_pred_${name}.txt -p > logs_${name}.txt

rank=16
name=lora${rank}

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --eval_interval 2000 \
    --save_interval 1000 \
    --lora_dim $rank \
    --lora_alpha $((2*rank)) \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/${name}_e2e \
    --random_seed 1 \
    --fp16

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/${name}_e2e/model.${iter}.pt \
    --platform local \
    --lora_dim ${rank} \
    --lora_alpha $((rank*2)) \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/${name}_e2e \
    --output_file predict.${iter}.b10p08r4.jsonl

python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/${name}_e2e/predict.${iter}.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref_${name}.txt \
    --output_pred_file e2e_pred_${name}.txt

python eval/e2e/measure_scores.py e2e_ref_${name}.txt e2e_pred_${name}.txt -p > logs_${name}.txt

rank=1024
name=vera${rank}

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.02 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --eval_interval 2000 \
    --save_interval 1000 \
    --lora_dim $rank \
    --lora_alpha $((2*rank)) \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/${name}_e2e \
    --fp16 \
    --random_seed 1 \
    --vera

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/${name}_e2e/model.${iter}.pt \
    --platform local \
    --lora_dim ${rank} \
    --lora_alpha $((rank*2)) \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/${name}_e2e \
    --output_file predict.${iter}.b10p08r4.jsonl \
    --random_seed 1 \
    --vera

python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/${name}_e2e/predict.${iter}.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref_${name}.txt \
    --output_pred_file e2e_pred_${name}.txt

python eval/e2e/measure_scores.py e2e_ref_${name}.txt e2e_pred_${name}.txt -p > logs_${name}.txt
