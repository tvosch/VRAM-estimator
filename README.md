## VRAM/GPU Memory estimator for LLMs
This repo estimates the required VRAM or commonly used GPU memory to train a large language model (LLM). 
Supports:
- [ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) stages.
- Providing HuggingFace hub repository id (example: meta-llama/Meta-Llama-3-8B)


## Getting Started
**In case you only use with your own estimations and numbers and no HuggingFace config is given, the prerequisities and installation can be skipped.**

### Prerequisites
- [transformers](https://huggingface.co/docs/transformers/en/index) (only necessary for automatic HuggingFace hub model parsing)


### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install transformers
```

### Usage
```bash
python vram_estimator_old.py --micro_batch_size 1 --num_gpus 1 --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3
```

When a repo_id is given, the argument parser values are overwritten


## Notes
Some models have been confirmed experimentally on their VRAM usage. See list below:

| Repo id/Model name                                 | Micro batch size | Number of GPUs | ZeRO stage | Gradient checkpointing | Estimated VRAM (per GPU)   | Actual VRAM (per GPU) |
|----------------------------------------------------|------------------|----------------|------------|------------------------| -------------------------- | --------------------- |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3 | 1                | 1              | 0          | False                  | 34.3 GB                    | 33.5GB                |
