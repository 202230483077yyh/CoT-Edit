# CoT-Edit

Official repository for **CoT-Edit**, a reinforcement learning framework for **code edit suggestion** with **chain-of-thought reasoning**.

## Introduction

Code edit suggestion—covering code modification, refactoring, and maintenance—is one of the most common activities in software development and has become a major focus of AI-powered coding tools. Existing approaches generally fall into two categories:

- **Instruction-based methods**, which translate explicit natural language instructions into code edits.
- **Pattern-based methods**, which learn from users' historical editing behaviors to generate more style-consistent and accurate suggestions.

Despite their promise, pattern-based methods still face two major challenges:

1. They struggle with edits that require **deep contextual reasoning**.
2. Their editing decisions often lack **interpretability**.

To address these limitations, we propose **CoT-Edit**, a reinforcement learning framework that enables large language models to discover **chain-of-thought (CoT) reasoning paths** for code editing **without requiring human-annotated CoT data**.

## Key Ideas

CoT-Edit is built around three main components:

- **Multi-step prompt templates**  
  Designed to support:
  - **analysis-guided code editing**
  - flexible switching between **CoT** and **non-CoT** inference modes

- **Edit-Aware Reward Modeling (EARM)**  
  A fine-grained, diff-based reward mechanism for more effective reinforcement learning in code editing tasks.

- **LoRA merging strategy**  
  A practical strategy we find effective for improving model generalization.

## Results

Experiments on an industrial dataset show that **CoT-Edit achieves 60.2% edit accuracy**, outperforming all strong baselines.  
In addition, **online A/B tests** further demonstrate its effectiveness in real-world production environments.

## Highlights

- Reinforcement learning for code edit suggestion
- Chain-of-thought discovery **without human-annotated CoT supervision**
- Fine-grained reward modeling tailored for code diffs
- Support for both **interpretable reasoning** and **efficient inference**
- Strong performance in both offline evaluation and online production testing

## Repository Overview

This repository contains the implementation of **CoT-Edit**, including:

- training and inference pipelines
- prompt templates for multi-step code editing
- reward modeling components for edit-aware optimization
- evaluation scripts and experimental settings

> More details on environment setup, data preparation, training, and evaluation will be provided in the corresponding sections.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{cotedit2026,
  title={CoT-Edit: Reinforcement Learning for Interpretable Code Edit Suggestion},
  author={Anonymous},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
