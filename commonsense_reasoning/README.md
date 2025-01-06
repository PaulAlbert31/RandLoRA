# Parameter-Efficient Fine-tuning for Large Language Models

This repository provides Python scripts for fine-tuning and evaluating large language models (LLMs) using RandLoRA and various parameter-efficient fine-tuning (PEFT) techniques. Code based on DoRA's [implementation](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning)

## Key Functionalities

* **Fine-tuning:** The `finetune.py` script allows you to fine-tune pre-trained LLMs on custom datasets. It supports different PEFT methods to reduce the number of trainable parameters.
* **Evaluation:** The `commonsense_evaluate.py` script evaluates the performance of fine-tuned models on common sense reasoning benchmark datasets.

## Supported Models

* LLaMA-3 (8B and 70B)
* Qwen2
* Phi3

## Supported PEFT Methods

* LoRA (Low-Rank Adaptation)
* RandLoRA (Randomized Low-Rank Adaptation)
* Vera
* DoRA

## Requirements

* Python 3.8+
* PyTorch
* Transformers library
* Peft library
* Datasets library
* Fire library
* Bitsandbytes (for 4-bit and 8-bit quantization)

You can install the necessary libraries using conda:

```bash
conda env create -f env.yml
conda activate randlora_cs
```

## Commonsense datasets

Follow instructions in the DoRA repository to install the commonsense datasets [here](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning#datasets).

## Usage

### Training script

The `base_X.sh` files provides examples of how to launch fine-tuning jobs. Here's a snippet illustrating the pattern for LoRA training on the 15k and 170k subset of the commonsense datasets:

```bash
for model in LLama3;do
    for n in 15 170;do
        r=32
        name="finetuned_result/${model}_lora${r}_${n}k"
        mkdir $name
        sh base_lora.sh $r $((2*r)) $name 0 $model $n
    done
done
```

This example demonstrates how to run the `base_lora.sh` script (provided in the repository) for different configurations of the LLaMA-3 model and dataset sizes. The `base_lora.sh` script further calls the `finetune.py` and `commonsense_evaluate.py` scripts with specific parameters.

### Fine-tuning

The `finetune.py` script provides various command-line arguments for configuring the fine-tuning process. Here's an example of how to run it (refer to `train_phoenix.txt` for more examples):

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model meta-llama/Meta-Llama-3-8B \
    --data_path commonsense_15k.json \
    --output_dir finetuned_result/LLama3_lora32_15k \
    --batch_size 16  --micro_batch_size 4 --num_epochs 1 \
    --learning_rate 1e-4 --cutoff_len 256 \
    --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 --lora_alpha 64 --use_gradient_checkpointing
```

**Explanation of key arguments:**

* `--base_model`: The name or path of the pre-trained language model.
* `--data_path`: Path to the training data in JSON format.
* `--output_dir`: Directory to save the fine-tuned model.
* `--batch_size`: The overall batch size.
* `--micro_batch_size`: The batch size per GPU.
* `--num_epochs`: Number of training epochs.
* `--learning_rate`: Learning rate for the optimizer.
* `--cutoff_len`: Maximum sequence length.
* `--adapter_name`: The PEFT method to use (e.g., `lora`, `randlora`, `vera`, `dora`).
* `--target_modules`: The modules in the model to apply the PEFT method to.
* `--lora_r`: The rank of the LoRA adapters.
* `--lora_alpha`: The scaling factor for the LoRA adapters.
* `--use_gradient_checkpointing`: Enables gradient checkpointing to reduce memory usage.
* `--load_4bit`: Enables quantized training QLoRA style.

### Evaluation

The `commonsense_evaluate.py` script evaluates the fine-tuned model on various common sense reasoning datasets. Here's an example of how to run it:

```bash
CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLama3-8B \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model meta-llama/Meta-Llama-3-8B \
    --batch_size 1 \
    --lora_weights finetuned_result/LLama3_lora32_15k
```

**Explanation of key arguments:**

* `--model`: The name of the model architecture (e.g., `LLaMA3-8B`).
* `--adapter`: The PEFT method used during fine-tuning (e.g., `LoRA`).
* `--dataset`: The name of the evaluation dataset (e.g., `openbookqa`, `ARC-Challenge`, etc.).
* `--base_model`: The name or path of the pre-trained language model.
* `--batch_size`: The batch size for evaluation.
* `--lora_weights`: Path to the directory containing the fine-tuned adapter weights.

The evaluation script will output the accuracy on the specified dataset and save the detailed results in the `experiment/` directory.


## Parsing experiment results

The final results can be obtained when finetuning has completed. Use the

```bash
python parse_exp_all.py finetuned_result/LLama3_lora32_15k
```

command to parse.

**Note:** Adapt the paths and model names according to your setup.
