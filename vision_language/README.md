# Fine-tuning Vision and Vision language Models

This repository provides a PyTorch implementation for fine-tuning vision models, particularly CLIP, using RandLoRA and other parameter-efficient fine-tuning (PEFT) methods. It includes finetuning on full datasets and few-shot adaptation

## Key Features

* **Parameter-Efficient Fine-tuning (PEFT):** Supports several PEFT techniques including:
    * **LoRA:**  Fine-tunes by adding low-rank matrices to existing weights.
    * **VeRA:**  Uses randomly initialized trainable matrices.
    * **NoLA:** A method likely inspired by LoRA for few-shot learning.
    * **RandLoRA:**  Experiments with random combinations of LoRA layers.
* **Full Fine-tuning:**  Option to fine-tune the entire model.
* **Few-Shot Learning:** Designed to work effectively with limited training data, allowing specification of the number of samples per class.
* **Integration with Accelerate:** Leverages the `accelerate` library for mixed-precision training and easy distributed training.
* **Comprehensive Logging:** Tracks training progress, loss, learning rate, and memory usage. Results are saved to text files for easy analysis.
* **Support for Multiple Datasets:** Easily configurable to train on various image classification datasets.

## Installation
   
Install dependencies: It is recommended to create a virtual environment.

```sh
conda env create -f randlora_vl.yml
conda activate randlora_vl
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Datasets
We use the same datasets as in Zhang et al. Instructions on how to install are available [here](https://github.com/fredzzhang/atlas/blob/main/DATASETS.md).
Remember to adjust the data-location argument in args.py to point to your dataset directory.

## Usage

The training script is src/finetune.py. You can configure the training process using command-line arguments defined in src/args.py.

### Basic Usage:
```sh
accelerate launch src/finetune.py --model <model_name> --train-dataset <dataset_name> --data-ratio <num_samples_per_class> --param-type <peft_method>
```

### Key arguments:

* --model: The name of the pre-trained model to use (e.g., ViT-B-32).

* --train-dataset: The name of the dataset to fine-tune on (e.g., CIFAR10).

* --data-ratio: The number of training samples per class (int) or use a float between 0.0 and 1.0 for a fraction of the dataset (add '.' for fractions).

* --param-type: The PEFT method to use (randlora, lora, vera, nola). Omit for full finetuning.
 
* --rank: The rank for NoLA, LoRA, VeRA, or RandLoRA.

* --lr: The learning rate.

* --batch-size: The batch size for training.

* --epochs: The number of training epochs.

* --seed: Random seed for reproducibility.

* --fname: Directory to save results (creates subdirectories for data_ratio and seed).

* --full-clip: Fine-tune both the vision and language encoders of CLIP (vision only by default).


Refer to train.sh for example usage.


### Customization

You can customize the training process by modifying the command-line arguments when running finetune.py. For example, you can change the learning rate, batch size, number of epochs, or the specific PEFT method used.


## Results

Training results, including training time, testing time, number of parameters, memory usage, and accuracy, are saved in the results/ directory. Each experiment will have its own subdirectory based on the model name, data ratio, and random seed. The results are stored in a results.txt file within these subdirectories.

