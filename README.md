# RANDLORA: FULL RANK PARAMETER-EFFICIENT FINE-TUNING OF LARGE MODELS
This repository contains the official code for RandLoRA, including a hugginface [PEFT](https://github.com/huggingface/peft) unofficial integration.

A preprint of the paper is available here [preprint]()

## Abstract
> ...<br/> Low-Rank Adaptation (LoRA) and its variants have shown impressive results in reducing the number of trainable parameters and memory requirements of large
transformer networks while maintaining fine-tuning performance. However, the low-rank nature of the weight update inherently limits the representation power of
the fine-tuned model, potentially compromising performance on complex tasks. This raises a critical question: when a performance gap between LoRA and
standard fine-tuning is observed, is it due to the reduced number of trainable parameters or the rank deficiency? This paper aims to answer this question by introducing RandLoRA, a parameter-efficient method that performs full-rank updates
using a learned linear combinations of low-rank, non-trainable random matrices. Our method limits the number of trainable parameters by restricting optimization
to diagonal scaling matrices applied to the fixed random matrices. This allows us to effectively overcome low-rank limitations while maintaining low parame-
ter count and memory usage during training. Through extensive experimentation across vision, language, and vision-language benchmarks, we systematically evaluate the limitations of LoRA and existing random basis methods. Our findings
reveal that full-rank updates are beneficial across vision and language tasks separately, but especially so for vision-language tasks, where RandLoRA significantly reduces—and sometimes eliminates—the performance gap between
standard finetuning and LoRA, demonstrating its efficacy

## Dino and CLIP results

## Langugage
The [commonsense_reasoning](commonsense_reasoning/) directory contains code to reproduce the paper's results.
### Requirements
#### Packages
The requirements are different than the vision experiments. Within the [commonsense_reasoning](commonsense_reasoning/) directory run
```
conda install randlora_llm.yml
conda activate randlora_llm
```

#### Datasets
Download commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and place them in the `commonsense_reasoning/datasets` folder.
The finetuning `commonsense_{15/170}k.json` datasets are also downloaded from the [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/ft-training_set) github repository. Place the finetuning json files in the [commonsense_reasoning](commonsense_reasoning/) directory.

## Training
The file [train.sh](commonsense_reasoning/train.sh) contains code for training multiple configurations of RandLoRA of Qwen2/Phi3/LLama3 using the hugginface PEFT and transformer libraries.
This codebase support quantized finetuning as in PEFT by passing `--load-{8/4}bit` to the [finetune.py](commonsense_reasoning/finetune.py) script.
We additionally support sparse random bases as described in [(Bingam et al)](https://cs-people.bu.edu/evimaria/cs565/kdd-rp.pdf) and [(Li et al)](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf). These are only provided as a proof of concept and do not provide any actual speedup or memory savings.

## Results


## Citation
If your work was relevant to your research, consider citing the paper

```bibtex
@article{2024_arxiv_RandLoRA,
  title={RandLoRA: full rank parameter-efficient fine-tuning of large models},
  author={Albert, Paul and Zhang, Frederic Z and Rodriguez-Opazo, Cristian and Saratchandran, Hemanth and Hengel, Anton van den and Abbasnejad, Ehsan},
  journal=,
  year={2024}
}
```