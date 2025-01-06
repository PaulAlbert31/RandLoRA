#FacebookAI/roberta-base
seq=128
bs=64
name=''
randlora_r=(64)

for model in FacebookAI/roberta-base FacebookAI/roberta-large;do
    for seed in 1 2 3 4 5;do
	for TASK_NAME in sst2 mrpc cola qnli rte stsb;do
	    #RandLoRA
	    for r in "${randlora_r[@]}";do
		python run_glue.py \
		       --model_name_or_path ${model} \
		       --task_name $TASK_NAME \
		       --do_train \
		       --do_eval \
		       --max_seq_length $seq \
		       --per_device_train_batch_size $bs \
		       --learning_rate 1e-4 \
		       --num_train_epochs 10 \
		       --output_dir finetuned_results/randlora${r}${name}/${seed}/${TASK_NAME}/ \
		       --fp16 \
		       --adapter_name randlora \
		       --rank $r \
		       --seed ${seed}
	    done
	    #LoRA
	    for r in 4;do 
		python run_glue.py \
		       --model_name_or_path ${model} \
		       --task_name $TASK_NAME \
		       --do_train \
		       --do_eval \
		       --max_seq_length $seq \
		       --per_device_train_batch_size $bs \
		       --learning_rate 1e-4 \
		       --num_train_epochs 10 \
		       --output_dir finetuned_results/lora${r}${name}/${seed}/${TASK_NAME}/ \
		       --fp16 \
		       --adapter_name lora \
		       --rank $r \
		       --seed ${seed}
	    done
	    #VeRA
	    for r in 256;do 
		python run_glue.py \
		       --model_name_or_path ${model} \
		       --task_name $TASK_NAME \
		       --do_train \
		       --do_eval \
		       --max_seq_length $seq \
		       --per_device_train_batch_size $bs \
		       --learning_rate 1e-2 \
		       --num_train_epochs 10 \
		       --output_dir finetuned_results/vera${r}${name}/${seed}/${TASK_NAME}/ \
		       --fp16 \
		       --adapter_name vera \
		       --rank $r \
		       --seed ${seed}
	    done
	done
    done
    #FacebookAI/roberta-large
    seq=128
    bs=64
    name='_large'
    randlora_r=(100)
done
