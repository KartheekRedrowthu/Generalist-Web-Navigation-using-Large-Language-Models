# Generalist Web Navigation using Large Language Models
### Implementation and Evaluation of the Mind2Web Web Agent Pipeline

**B-Tech Project** | Indian Institute of Technology Bhubaneswar
**Author:** Redrowthu Kartheek (22CS01012)
**Guide:** Dr. Shreya Ghosh
**Department:** Computer Science and Engineering

---

## Table of Contents

- [About Mind2Web](#about-mind2web)
- [Dataset](#dataset)
- [Pipeline Overview](#pipeline-overview)
- [Repository Structure](#repository-structure)
- [Hosted Resources on HuggingFace](#hosted-resources-on-huggingface)
- [Setup and Installation](#setup-and-installation)
- [Experiments](#experiments)
  - [Candidate Generation](#candidate-generation)
  - [Action Prediction](#action-prediction)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [References](#references)

---

## About Mind2Web

[Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) is the first large-scale dataset designed specifically for training and evaluating **general-purpose web-navigation agents**. Unlike traditional automation datasets that rely on synthetic or controlled environments, Mind2Web captures interactions on real-world websites, making it far more realistic and challenging.

Each task consists of a natural language instruction paired with a sequence of human-demonstrated actions — including clicking elements, entering text, selecting options, and navigating across multiple pages — providing rich supervision for grounded web behavior.

The Mind2Web pipeline decomposes web navigation into two sequential stages:

1. **Candidate Generation** — A ranking model (DeBERTa-v3-base) identifies the most relevant DOM elements from thousands of nodes on a webpage.
2. **Action Prediction (MindAct)** — Given the ranked candidates and instruction, the model predicts the correct operation (`CLICK`, `TYPE`, or `SELECT`) and the exact target element.

---

## Dataset

### Coverage

| Property | Value |
|---|---|
| Real-world websites | 137 |
| Distinct domains | 31 |
| Task trajectories | 2,000+ |
| Action types | CLICK, TYPE, SELECT |

Domains include: Travel, Shopping, Entertainment, Finance, Info portals, and more.

### Evaluation Splits

| Split | Description |
|---|---|
| **Cross-Task** | Unseen tasks on websites seen during training |
| **Cross-Website** | Entirely unseen websites within known domains |
| **Cross-Domain** | Webpages from completely unseen domains |

### Key Dataset Fields

**Task-level:** `annotation_id`, `website`, `domain`, `confirmed_task`, `action_reprs`

**Step-level:** `action_uid`, `raw_html`, `cleaned_html`, `operation`, `op`, `value`

**Element-level:** `pos_candidates`, `neg_candidates`, `tag`, `backend_node_id`, `attributes`, `is_original_target`, `is_top_level_target`

### Downloading the Dataset

The original Mind2Web dataset can be found at:
- HuggingFace: https://huggingface.co/datasets/osunlp/Mind2Web
- GitHub: https://github.com/OSU-NLP-Group/Mind2Web

---

## Pipeline Overview

```
Natural Language Instruction
         │
         ▼
┌─────────────────────────┐
│   DOM Processing        │
│  (Cleaned HTML / DOM)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 1:               │
│  Candidate Generation   │  ◄── DeBERTa-v3-base (fine-tuned)
│  (Top-K DOM Elements)   │
└────────────┬────────────┘
             │  Top-K candidates (K = 20–50)
             ▼
┌─────────────────────────┐
│  Stage 2:               │
│  Action Prediction      │  ◄── MindAct (Flan-T5-large)
│  (MindAct)              │
└────────────┬────────────┘
             │
             ▼
   Predicted Action + Target Element
   (CLICK / TYPE / SELECT)
```

---

## Repository Structure

```
.
├── candidate_generation/
│   ├── train.py                  # Fine-tuning script for DeBERTa-v3-base
│   ├── evaluate.py               # Evaluation script for candidate generation
│   ├── data_utils.py             # Dataset loading and preprocessing
│   └── config.py                 # Training configuration
│
├── action_prediction/
│   ├── train.py                  # MindAct training script
│   ├── evaluate.py               # Evaluation script for action prediction
│   ├── data_utils.py             # Data preparation for MindAct
│   └── config.py                 # Training configuration
│
├── data/
│   └── mind2web/                 # Place dataset files here
│
├── outputs/
│   ├── candidate_gen/            # Saved models and score files
│   └── action_pred/              # Saved action prediction models
│
└── README.md
```

---

## Hosted Resources on HuggingFace

The following resources are hosted on HuggingFace for this project. Note: only **training** score files are provided for candidate generation (not test).

### Dataset

| Resource | URL |
|---|---|
| Mind2Web Dataset (original) | `https://huggingface.co/datasets/osunlp/Mind2Web` |
| Project Dataset (hosted) | `<!-- ADD YOUR HF DATASET URL HERE -->` |

### Models

| Model | Description | URL |
|---|---|---|
| Candidate Generation Model | Fine-tuned DeBERTa-v3-base | `<!-- ADD YOUR HF MODEL URL HERE -->` |
| Action Prediction Model | Fine-tuned Flan-T5-large (MindAct) | `<!-- ADD YOUR HF MODEL URL HERE -->` |

### Score Files (Candidate Generation — Train Only)

| Split | URL |
|---|---|
| Train Scores | `<!-- ADD YOUR HF SCORE FILE URL HERE -->` |

> **Note:** Test score files for candidate generation are not hosted. Run evaluation locally using the instructions below to regenerate them.

### Loading Hosted Models

```python
from transformers import AutoModel, AutoTokenizer

# Candidate Generation Model
tokenizer = AutoTokenizer.from_pretrained("<!-- YOUR HF MODEL REPO -->")
model = AutoModel.from_pretrained("<!-- YOUR HF MODEL REPO -->")

# Action Prediction Model
from transformers import AutoModelForSeq2SeqLM
action_model = AutoModelForSeq2SeqLM.from_pretrained("<!-- YOUR HF ACTION MODEL REPO -->")
```

---

## Setup and Installation

### Requirements

```bash
pip install torch transformers datasets accelerate
pip install selenium playwright   # optional, for live webpage interaction
```

### Clone the Repository

```bash
# ADD YOUR REPO CLONE COMMAND HERE
# git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
# cd YOUR_REPO
```

### Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Experiments

### Candidate Generation

The candidate generation stage uses **DeBERTa-v3-base** to rank DOM elements by relevance to the user instruction. Fine-tuning is performed as a ranking-based classification task.

#### Training (Fine-Tuning)

```bash
# ADD CANDIDATE GENERATION TRAINING COMMAND HERE
# Example:
# python candidate_generation/train.py \
#     --model_name microsoft/deberta-v3-base \
#     --dataset_path data/mind2web/ \
#     --output_dir outputs/candidate_gen/ \
#     --epochs 5 \
#     --learning_rate 2e-5 \
#     --batch_size 32
```

#### Evaluation

```bash
# ADD CANDIDATE GENERATION EVALUATION COMMAND HERE
# Example:
# python candidate_generation/evaluate.py \
#     --model_path outputs/candidate_gen/ \
#     --split cross_task          # Options: cross_task, cross_website, cross_domain
```

#### Loading Train Score Files from HuggingFace

```bash
# ADD COMMAND TO DOWNLOAD SCORE FILES FROM HF HERE
# Example:
# huggingface-cli download <!-- YOUR HF REPO --> train_scores.json --local-dir outputs/candidate_gen/
```

---

### Action Prediction

The action prediction stage uses **MindAct** (Flan-T5-large) to predict the operation type and target element from ranked candidates.

Six experiments were conducted to study the effect of different training strategies:

| Experiment | Description |
|---|---|
| **Exp 1** | Baseline — default config, original candidate generation outputs |
| **Exp 2** | Fine-tuned candidate generation + default action prediction |
| **Exp 3** | Default candidate generation + modified action prediction training |
| **Exp 4** | Fine-tuned candidate generation + optimized action prediction |
| **Exp 5** | Increased training epochs |
| **Exp 6** | Modified learning rate schedule |

#### Experiment 1 — Baseline

```bash
# ADD EXPERIMENT 1 COMMAND HERE
# Example:
# python action_prediction/train.py \
#     --candidate_scores outputs/candidate_gen/baseline_scores/ \
#     --output_dir outputs/action_pred/exp1/ \
#     --config configs/baseline.json
```

#### Experiment 2 — Fine-Tuned Candidate Generation

```bash
# ADD EXPERIMENT 2 COMMAND HERE
# Example:
# python action_prediction/train.py \
#     --candidate_scores outputs/candidate_gen/finetuned_scores/ \
#     --output_dir outputs/action_pred/exp2/ \
#     --config configs/baseline.json
```

#### Experiment 3 — Modified Action Prediction Training

```bash
# ADD EXPERIMENT 3 COMMAND HERE
# Example:
# python action_prediction/train.py \
#     --candidate_scores outputs/candidate_gen/baseline_scores/ \
#     --output_dir outputs/action_pred/exp3/ \
#     --config configs/modified_action.json
```

#### Experiment 4 — Combined Candidate and Action Optimization

```bash
# ADD EXPERIMENT 4 COMMAND HERE
# Example:
# python action_prediction/train.py \
#     --candidate_scores outputs/candidate_gen/finetuned_scores/ \
#     --output_dir outputs/action_pred/exp4/ \
#     --config configs/modified_action.json
```

#### Experiment 5 — Change in Epochs

```bash
# ADD EXPERIMENT 5 COMMAND HERE
# Example:
# python action_prediction/train.py \
#     --candidate_scores outputs/candidate_gen/finetuned_scores/ \
#     --output_dir outputs/action_pred/exp5/ \
#     --epochs 10 \
#     --config configs/baseline.json
```

#### Experiment 6 — Change in Learning Rate Schedule

```bash
# ADD EXPERIMENT 6 COMMAND HERE
# Example:
# python action_prediction/train.py \
#     --candidate_scores outputs/candidate_gen/finetuned_scores/ \
#     --output_dir outputs/action_pred/exp6/ \
#     --lr_scheduler cosine \
#     --config configs/baseline.json
```

#### Running Evaluation on All Experiments

```bash
# ADD EVALUATION COMMAND HERE
# Example:
# for exp in exp1 exp2 exp3 exp4 exp5 exp6; do
#     python action_prediction/evaluate.py \
#         --model_path outputs/action_pred/$exp/ \
#         --split cross_task cross_website cross_domain
# done
```

---

## Evaluation Metrics

### Candidate Generation

| Metric | Description |
|---|---|
| **Accuracy@1** | Fraction of samples where correct element is ranked first |
| **MRR** | Mean Reciprocal Rank of the correct element |
| **Recall@K** | Fraction of samples where correct element appears in top-K (K = 3, 5, 10, 20, 50, 100) |

### Action Prediction

| Metric | Description |
|---|---|
| **Element Accuracy** | Fraction of steps where the correct DOM element is selected |
| **Action F1** | Token-level F1 over predicted operation and parameters |
| **Step Accuracy** | Fraction of steps where both operation type and target element are correct |

---

## Results Summary

### Candidate Generation — Effect of Fine-Tuning

| Metric | Before FT (Cross-Task) | After FT (Cross-Task) |
|---|---|---|
| MRR | 0.4017 | 0.4191 |
| Accuracy@1 | 0.2468 | 0.2586 |
| Recall@5 | 0.6032 | 0.6206 |
| Recall@10 | 0.7387 | 0.7542 |

### Action Prediction — All Experiments

| Experiment | Split | Element Acc | Action F1 | Step Acc |
|---|---|---|---|---|
| Exp 1 (Baseline) | Cross-Task | 0.5072 | 0.7483 | 0.4633 |
| Exp 2 (FT Cand Gen) | Cross-Task | 0.5066 | 0.7468 | 0.4660 |
| Exp 3 (Modified AP) | Cross-Task | 0.4580 | 0.7643 | 0.4211 |
| Exp 4 (Combined) | Cross-Task | 0.4582 | 0.7636 | 0.4201 |
| Exp 5 (More Epochs) | Cross-Task | 0.4711 | **0.7737** | 0.4314 |
| Exp 6 (LR Schedule) | Cross-Task | 0.4702 | 0.7590 | 0.4261 |
| Exp 5 (More Epochs) | Cross-Domain | 0.3725 | 0.6878 | 0.3419 |
| Exp 6 (LR Schedule) | Cross-Domain | 0.3719 | **0.6966** | 0.3351 |

> **Best Cross-Task Action F1:** Experiment 5 (0.7737)
> **Best Cross-Domain Action F1:** Experiment 6 (0.6966)

---

## References

- Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., & Su, Y. (2023). **MIND2WEB: Towards a Generalist Agent for the Web.** NeurIPS 2023.
- Pahuja, V. et al. (2025). **Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents.** ACL 2025 Findings.
- [Mind2Web GitHub Repository](https://github.com/OSU-NLP-Group/Mind2Web)
- [MindAct CandidateGeneration deberta-v3-base Model](https://huggingface.co/osunlp/MindAct_CandidateGeneration_deberta-v3-base)
- [MindAct ActionPrediction flan-t5-large Model](https://huggingface.co/osunlp/MindAct_ActionPrediction_flan-t5-large)
- [Puppeteer Headless Chrome Node.js API](https://pptr.dev/)

---

*B-Tech Project, IIT Bhubaneswar, May 2026*
