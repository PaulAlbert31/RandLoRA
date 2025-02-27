export PYTHONPATH="$PYTHONPATH:$PWD"
#Training CLIP on one dataset
dataset=UCF101
ratio=1.0
model=ViT-B-32

accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_RandLoRA6 --rand-lora --rank 6 --data-ratio $ratio --param-type 'randlora' --full-clip --dataset ${dataset} --save-weights
accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_LoRA32 --rand-lora --rank 32 --data-ratio $ratio --param-type 'lora' --full-clip --dataset ${dataset} --save-weights
accelerate launch src/finetune.py --model=$model --seed 1 --fname results/${model}_FT --data-ratio $ratio --full-clip --dataset ${dataset} --save-weights --lr 1e-5

accelerate launch loss_barrier.py --model=${model} --seed 1 --data-ratio 0.01 --full-clip --dataset ${dataset} --vis-models results/${model}_RandLoRA6/${ratio}/1/model_${dataset}.pth results/${model}_FT/${ratio}/1/model_${dataset}.pth results/${model}_LoRA32/${ratio}/1/model_${dataset}.pth
