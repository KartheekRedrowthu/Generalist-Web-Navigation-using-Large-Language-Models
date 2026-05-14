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
├── src/
│   ├── action_prediction/
│   │   ├── conf/                        # Configuration files
│   │   ├── __init__.py
│   │   ├── dataloader.py                # Dataset loading utilities
│   │   ├── evaluate.py                  # Evaluation script
│   │   ├── evaluate_llm.py              # LLM-based evaluation script
│   │   ├── llm_prompt.json              # Prompt templates for LLM evaluation
│   │   ├── metric.py                    # Evaluation metrics
│   │   ├── model.py                     # MindAct model definition
│   │   └── train.py                     # Action prediction training script
│   │
│   ├── candidate_generation/
│   │   ├── conf/                        # Configuration files
│   │   ├── __init__.py
│   │   ├── dataloader.py                # Dataset loading utilities
│   │   ├── evaluate.py                  # Evaluation script
│   │   ├── metric.py                    # Evaluation metrics
│   │   ├── model.py                     # Candidate generation model
│   │   └── train.py                     # Candidate generation training script
│   │
│   └── data_utils/                      # Shared preprocessing utilities
│
├── Action_Pred_Results/                 # Action prediction outputs and scores
├── Cand_Gen_Results/                    # Candidate generation outputs and scores
│
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
└── data_inspector.ipynb                 # Dataset inspection notebook
```

---

## Hosted Resources on HuggingFace

The following resources are hosted on HuggingFace for this project. Note: only **training** score files are provided for candidate generation (not test).

### Models

#### Candidate Generation Models

| Model | Description | URL |
|---|---|---|
| MindAct Candidate Generation (DeBERTa-v3-base) | Original MindAct candidate generation model based on DeBERTa-v3-base | https://huggingface.co/KartheekRedrowthu/Before_FT_Cand_Gen_Model |
| Fine-tuned Candidate Generation Model | Fine-tuned candidate generation model trained on Mind2Web | https://huggingface.co/KartheekRedrowthu/After_FT_Cand_Gen_Model |

#### Action Prediction Models

| Model | Description | URL |
|---|---|---|
| MindAct Action Prediction (Flan-T5-large) | Original MindAct action prediction model based on Flan-T5-large | https://huggingface.co/KartheekRedrowthu/Before_FT_Act_Pred_Model |
| Action Prediction Model with LR Change | Fine-tuned action prediction model with modified learning rate settings | https://huggingface.co/KartheekRedrowthu/Act_Pred_Model_with_lr_change |
| Action Prediction Model with Epoch Change | Fine-tuned action prediction model with modified epoch settings | https://huggingface.co/KartheekRedrowthu/Act_Pred_Model_with_epoch_change |
| Fine-tuned Action Prediction Model | Fine-tuned action prediction model trained on Mind2Web | https://huggingface.co/KartheekRedrowthu/After_FT_Act_Pred_Model |

### Score Files (Candidate Generation — Train Only)

| Split | URL |
|---|---|
| Before FT Train Scores | https://huggingface.co/datasets/KartheekRedrowthu/before_ft_cand_train_results |
| After FT Train Scores | https://huggingface.co/datasets/KartheekRedrowthu/after_ft_cand_gen_train_results |
> **Note:** Test score files for candidate generation are not hosted. You can Find them in Cand_Gen_Results.

## Download Model

### Using Transformers

```python
from transformers import AutoTokenizer, AutoModel

model_name = "KartheekRedrowthu/After_FT_Cand_Gen_Model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### Using Git LFS

```bash
git lfs install
git clone https://huggingface.co/KartheekRedrowthu/After_FT_Cand_Gen_Model
```
## Setup and Installation

### Clone the Repository

```bash
 git clone https://github.com/KartheekRedrowthu/Generalist-Web-Navigation-using-Large-Language-Models.git
 cd Generalist-Web-Navigation-using-Large-Language-Models
```

### Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```
> **Note:** Make sure you use Python 3.10 Version.
---

## Experiments

### Candidate Generation

The candidate generation stage uses **DeBERTa-v3-base** to rank DOM elements by relevance to the user instruction. Fine-tuning is performed as a ranking-based classification task.

#### Training (Fine-Tuning)

```bash
python src/candidate_generation/train.py \
    model=deberta-v3-base \
    hydra.run.dir= #Your Output folder to save model#
```

#### Evaluation
Run the below command for every Split in the Data (i.e test task-0,1,2 ; test_website-0,1 ; test_domain-0,1,..,9)

```bash
python src/candidate_generation/evaluate.py \
    --model_path /path/to/model_checkpoint \
    --data_path /path/to/dataset \
    --split_file /path/to/dataset/test_domain/test_domain_4.json \
    --output_dir /path/to/output_folder/test_domain/test_domain_4
```

## Loading Train Score Files from Hugging Face

```bash
pip install huggingface_hub

huggingface-cli download KartheekRedrowthu/before_ft_cand_train_results \
scores_train.pkl \
--local-dir outputs/candidate_gen/
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

You can find all the Models mentioned in the above links 

For training your Model, Use the following command:
```bash
torchrun --nproc-per-node 1 --master_port=8004 \
    src/action_prediction/train.py \
    model=flan-t5-large \
    data.data_path=/path/to/mind2web_dataset \
    data.score_file=/path/to/candidate_generation_scores/scores_train.pkl \
    train.per_device_train_batch_size=8 \
    train.gradient_accumulation_steps=1 \
    train.fsdp=False \
    train.num_gpus=1 \
    train.epoch=5 \
    run_id="full" \
    hydra.run.dir=/path/to/action_prediction_model_output
```

#### Running Evaluation 
> **Note:** Ensure that the dataset split specified inside `evaluate.py` matches the split provided in the evaluation command.

> **Note:** To modify training hyperparameters such as the number of epochs or learning rate schedule, update the corresponding settings in the `config.yaml` file inside the `action_prediction` module.


```bash
python src/action_prediction/evaluate.py \
    +model_path=/path/to/action_prediction_model \
    +output_path=/path/to/evaluation_results/test_domain/test_domain_9 \
    +top_k=50 \
    data.test_split_files.test_domain=/path/to/dataset/test_domain/test_domain_9.json \
    data.score_file=/path/to/candidate_generation_results/test_domain/test_domain_9/scores_test_domain.pkl
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

- [Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., & Su, Y. (2023). **MIND2WEB: Towards a Generalist Agent for the Web.** NeurIPS 2023.](https://arxiv.org/pdf/2306.06070)
- [Pahuja, V. et al. (2025). **Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents.** ACL 2025 Findings.](https://arxiv.org/pdf/2502.11357)
- [Mind2Web GitHub Repository](https://github.com/OSU-NLP-Group/Mind2Web)
- [MindAct CandidateGeneration deberta-v3-base Model](https://huggingface.co/osunlp/MindAct_CandidateGeneration_deberta-v3-base)
- [MindAct ActionPrediction flan-t5-large Model](https://huggingface.co/osunlp/MindAct_ActionPrediction_flan-t5-large)


---

*B-Tech Project, IIT Bhubaneswar, May 2026*

