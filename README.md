# DDI-EviText

This repository contains the pytorch implementation of **Evidence-grounded consensus reasoning enables drug–drug interaction prediction for emerging drugs and rare interactions**. 

![ramework for DDI-EviText.](DDI-EviText.png)

**Framework for DDI-EviText.** The diagram outlines the key modules: Biological Evidence Retrieval and Unification, Tail-Aware and Pharmacology-Guided Training, and Consensus-Driven Consistent Inference.

## Setup

The following packages are required for running GenART. Compatibility is guaranteed with the specified versions:
- **torch**: `2.7.1`
- **torchvision**: `0.22.1`
- **transformers**: `4.57.6`
- **flash-attn**: `2.8.3`


## Datasets

We provide DrugBank benchmark datasets and TWOSIDES benchmark datasets for pretrained models. You can access them via the following Google Drive links:

[Benchmark datasets](https://drive.google.com/file/d/1WSPYbNdngCTECRYD2pTF-q-kUruR8HBm/view?usp=drive_link)

## Usage
We provide code for Supervised Fine-Tuning and Inference, which can be found in `model_sft.py` and `vLLM_inference.py`.

- **Training**: LoRA + FSDP fine-tuning for Qwen3 LLM.
- **Inference**: vLLM inference with optional LoRA adapter.

The execution commands are as follows:

## Training

```shell
# Multi-GPU Supervised Fine-Tuning
accelerate config --config_file "fsdp_config.yaml" train/model_sft.py

> Note: FSDP requires an `accelerate config`. Run `accelerate config` once to generate your config for your cluster/GPU setup.

```

## Inference

```shell
python inference/vLLM_inference.py \
  --backend vllm \
  --model-name "Qwen/Qwen3-4B-Thinking-2507" \
  --lora-path ./checkpoints/lora_adapter \
  --drugs-file ./data/drugs_info.csv \
  --input-file ./data/DrugBank/S1/test_sft.csv \
  --kg-path-file ./data/DrugBank/S1/test_kg.json \
  --output-file ./outputs/test_results.csv
```
