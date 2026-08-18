# DA6401 Assignment 3: Transformer for Machine Translation

## W&B Report Links

| Question | Topic | Report |
|---|---|---|
| 2.1 | Noam Scheduler vs Fixed Learning Rate | [Open report](https://wandb.ai/v-eswar2005-massachusetts-institute-of-technology/DA6401_assignement3_EE23B085/reports/2-1-The-Necessity-of-the-Noam-Scheduler-10-Marks---VmlldzoxNjg4MzQzNw?accessToken=bngme9scl1v7sbod5uycut3f9gqxgysiyv4xlexpoxsyuef8qz2gesxb2cwqys8d) |
| 2.2 | Attention Scaling Factor Ablation | [Open report](https://wandb.ai/v-eswar2005-massachusetts-institute-of-technology/DA6401_assignement3_EE23B085/reports/2-2-Ablation-The-Scaling-Factor-1-dk-10-Marks---VmlldzoxNjg4MzY0Nw?accessToken=1zjpt7vtza3je5e9bc8t70m4x61kgyqk3wkryo5gflqk9xphfulr68fqpxij7fhz) |
| 2.3 | Attention Rollout and Head Specialization | [Open report](https://wandb.ai/v-eswar2005-massachusetts-institute-of-technology/DA6401_assignement3_EE23B085/reports/Question-2-3-Attention-Rollout-and-Head-Specialization--VmlldzoxNjg4MzgzNg?accessToken=wlp6gsrbt7ejafl9m4jsm7y12cfwpouf5ua6drgkmcsf4xigfn1p12gcynb1igg2) |
| 2.4 | Sinusoidal vs Learned Positional Embeddings | [Open report](https://wandb.ai/v-eswar2005-massachusetts-institute-of-technology/DA6401_assignement3_EE23B085/reports/2-4-Positional-Encoding-vs-Learned-Embeddings-10-Marks---VmlldzoxNjg4NDEwNQ?accessToken=nk17lrd0ulfpf5do8gd9h111965xe7yh3y6rke4knu2jox2z9glzfp4klmk0t88o) |
| 2.5 | Decoder Sensitivity to Label Smoothing | [Open report](https://wandb.ai/v-eswar2005-massachusetts-institute-of-technology/DA6401_assignement3_EE23B085/reports/2-5-Decoder-Sensitivity-Label-Smoothing-10-Marks---VmlldzoxNjg4NDE1OA?accessToken=3xwheekzo1kkrcqsyf5d9ti5z62kim5fg6jmnb7jtp03vgj91yhagp5i562vqty5) |


GitHub repository: [da6401_assignement3](https://github.com/ee23b067-cmd/da6401_assignement3/tree/main)

## Project Overview

This repository implements a Transformer-based neural machine translation system for the DA6401 Introduction to Deep Learning Assignment 3. The model follows the core ideas from **"Attention Is All You Need"** and is trained for **German to English translation** on the Multi30k dataset.

The project contains both the implementation and the experiment analysis required for the assignment:

- Scaled dot-product attention and multi-head attention implemented from scratch.
- Transformer encoder and decoder stacks with residual connections, layer normalization, and feed-forward blocks.
- Sinusoidal positional encoding, with an ablation against learned positional embeddings.
- Noam learning-rate scheduler with warmup and inverse-square-root decay.
- Label smoothing and standard cross-entropy comparisons.
- Greedy decoding and BLEU evaluation.
- W&B reports for scheduler behavior, attention scaling, head specialization, positional encoding, and label smoothing.

## Repository Structure

```text
DL_Assignement3/
├── dataset.py                     # Multi30k dataset loading, tokenization, and vocabulary utilities
├── lr_scheduler.py                # Noam learning-rate scheduler implementation
├── model.py                       # Transformer, attention, masks, positional encoding, inference helpers
├── train.py                       # Training loop, evaluation, checkpointing, BLEU computation
├── requirements.txt               # Python package requirements
├── ass_md.md                      # Assignment statement/reference
├── da-pa3-2.ipynb                 # Notebook used for experiments
├── checkpoints/                   # Saved checkpoints, created during training
└── wandb_report_files/
    ├── 2_1_submission.md          # Written analysis for Noam scheduler experiment
    ├── 2_2_submission.md          # Written analysis for attention scaling ablation
    ├── 2_3_submission.md          # Written analysis for attention rollout/head specialization
    ├── 2_4_submission.md          # Written analysis for positional encoding ablation
    ├── 2_5_submission.md          # Written analysis for label smoothing experiment
    └── all_questions_wandb_repot.ipynb
```

## What Was Implemented

### Transformer Architecture

The Transformer is implemented in PyTorch using basic modules such as `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and custom attention functions. The implementation includes:

- Scaled dot-product attention:

  ```text
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
  ```

- Multi-head attention without using `torch.nn.MultiheadAttention`.
- Padding masks for source/target sequences.
- Causal look-ahead masks for decoder self-attention.
- Encoder and decoder layers with attention, feed-forward networks, residual connections, and layer normalization.
- Greedy decoding for translation inference.

### Training and Optimization

The training pipeline includes the optimization choices required by the assignment:

- **Noam scheduler:** warmup followed by inverse-square-root learning-rate decay.
- **Label smoothing:** default smoothing value of `0.1`.
- **BLEU evaluation:** corpus-level BLEU for validation and test translation quality.
- **Checkpointing:** saves latest and best checkpoints for later evaluation/inference.

## W&B Experiments

The W&B reports document the main ablation studies for Assignment 3.

### 2.1 Noam Scheduler

Compares the Transformer trained with the Noam scheduler against a fixed learning rate. The report explains why warmup stabilizes early Transformer training and why the scheduled run converges better.

### 2.2 Attention Scaling Factor

Compares scaled dot-product attention with an unscaled version. The report focuses on query/key gradient norms and explains how large attention logits can saturate softmax and weaken useful gradients.

### 2.3 Attention Rollout and Head Specialization

Visualizes attention weights from the last encoder layer. The report identifies heads that behave like local-position trackers, phrase-anchor heads, and broader context aggregation heads, while also discussing head redundancy.

### 2.4 Positional Encoding vs Learned Embeddings

Compares sinusoidal positional encoding with `torch.nn.Embedding` learned positions. The report includes validation metric comparisons and explains why sinusoidal encoding can theoretically extrapolate to longer sequence lengths.

### 2.5 Label Smoothing

Compares `epsilon_ls = 0.1` with standard cross-entropy. The report discusses prediction confidence and explains why label smoothing regularizes the decoder by preventing over-confident softmax outputs.

## Setup

Create an environment and install the required packages:

```powershell
pip install -r requirements.txt
```

The project uses spaCy tokenization through the dataset utilities. If tokenizer models are missing in your environment, install the relevant spaCy language models before training.

## Training

Run the default training pipeline from the project root:

```powershell
python train.py
```

Useful command-line overrides include:

```powershell
python train.py --num-epochs 10
python train.py --lr-strategy noam
python train.py --learning-rate 1e-4
python train.py --label-smoothing 0.1
python train.py --positional-encoding sinusoidal
python train.py --positional-encoding learned
python train.py --no-attention-scaling
python train.py --resume-checkpoint checkpoints/latest_checkpoint.pt
```

Checkpoints are saved in `checkpoints/` by default, and the best checkpoint can be evaluated after training.

## Notes for Evaluation

The implementation is designed around the assignment/autograder requirements:

- `scaled_dot_product_attention` returns both attended outputs and attention weights.
- `MultiHeadAttention` is implemented manually.
- `PositionalEncoding` uses the sinusoidal formula and is stored as a non-trainable buffer.
- `make_src_mask` and `make_tgt_mask` support padding and causal masking.
- The saved checkpoint contains model weights and vocabulary assets needed for inference.

## Reference

Vaswani et al., **Attention Is All You Need**:  
https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
