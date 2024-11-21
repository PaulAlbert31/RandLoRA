export PYTHONPATH="$PYTHONPATH:$PWD"
#Training CLIP on 22 datasets
for model in ViT-B-32;do
    for ratio in 1 2 4 16 0.5 1.0;do
	accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_RandLoRA6 --rand-lora --rank 6 --data-ratio $ratio --param-type 'randlora' --full-clip
	accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_LoRA32 --rand-lora --rank 32 --data-ratio $ratio --param-type 'lora' --full-clip
	accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_VeRA256 --rand-lora --rank 256 --data-ratio $ratio --param-type 'vera' --full-clip --lr 1e-2
	accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_NoLA --rand-lora --rank 1 --data-ratio $ratio --param-type 'nola' --full-clip
	accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_FT --data-ratio $ratio --full-clip
    done
done
