# 🎙️ Speaker Verification & Gender Identification using ECAPA-TDNN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![SpeechBrain](https://img.shields.io/badge/SpeechBrain-ECAPA--TDNN-orange)](https://speechbrain.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-ai4bharat%2FLahaja-green)](https://huggingface.co/datasets/ai4bharat/Lahaja)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **Investigating how a speaker verification model trained on VoxCeleb spontaneously encodes gender as emergent structure — achieving 99.62% gender classification accuracy from frozen embeddings on Indian-accented speech.**

---

## 🔑 Key Results at a Glance

| Task | Result |
|------|--------|
| Speaker Verification (EER) | **8.2%** on Lahaja (cross-domain, Indian accent) |
| Verification AUC-ROC | **0.97** |
| Gender Classification Accuracy | **100%** (single split), **99.62% ± 0.19%** (5-fold CV) |
| Gender in PCA dimensions | **1 component** → 99.2% accuracy |
| Demo | **Live Gradio UI** with mic recording + threshold slider |

---

## 🧠 Core Finding

ECAPA-TDNN was trained **purely on speaker identity** (VoxCeleb1+2, 7000+ speakers) — it was **never given gender labels**. Yet gender information dominates its embedding space:

- The **1st principal component** of the 192-dim embedding encodes gender with **99.2% accuracy**
- A **logistic regression on frozen embeddings** achieves **100% gender classification**
- **5-fold cross-validation** confirms this is not overfitting: test accuracy = **99.62% ± 0.19%**

This emergent disentanglement — gender in one axis, speaker identity distributed across ~125 dimensions — reveals how deep speaker models learn hierarchically structured representations.

---

## 📁 Repository Structure

```
speaker-verification-ecapa/
│
├── notebooks/
│   └── speaker_verification_full.ipynb   # Complete pipeline — all tasks A.1 to A.4 + Bonus
│
├── src/
│   ├── dataset_construction.py           # Task A.1 — trial pair generation
│   ├── embedding_extraction.py           # Task A.2 — ECAPA-TDNN inference pipeline
│   ├── gender_separability.py            # Task A.3 — t-SNE, UMAP, PCA analysis
│   ├── gender_classifier.py              # Task A.4 — logistic regression + SVM + CV
│   └── utils.py                          # Shared helpers (audio loading, resampling)
│
├── demo/
│   └── app.py                            # Gradio UI — real-time verification + gender detection
│
├── results/
│   ├── plots/                            # t-SNE, UMAP, ROC curve, confusion matrix images
│   └── metrics/                          # Threshold sweep CSV, CV results
│
├── docs/
│   └── Speaker_Verification_Report.pdf   # Full technical report (15 pages)
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Tasks Overview

### Task A.1 — Dataset Construction
Built a **gender-balanced verification trial set** from scratch on [ai4bharat/Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja) — a Hindi multi-accent ASR benchmark with no pre-built speaker pairs.

- **600 trial pairs** (300 positive + 300 negative), `seed=42`
- **Stratified negative sampling**: 50% same-gender, 50% cross-gender
- Prevents the model from "cheating" by detecting gender instead of speaker identity
- Duplicate prevention via canonical key: `(min(idx_a, idx_b), max(idx_a, idx_b))`

**Bias discussion**: Cross-gender negatives inflate EER optimism (gender detection ≠ speaker verification). Accent bias also discussed — same-district negative stratification recommended for production.

---

### Task A.2 — Embedding Extraction & Speaker Verification
Extracted **192-dim ECAPA-TDNN embeddings** for all 1,064 utterances using SpeechBrain.

**Extraction pipeline:**
```
Audio → Mono → 16kHz resample → encode_batch() → (192,) numpy array
```
Unique-index caching: O(unique utterances) instead of O(pairs).

**Verification results:**

| Metric | Value |
|--------|-------|
| EER | **8.2%** |
| EER Threshold | 0.30 |
| AUC-ROC | **~0.97** |
| Mean same-speaker similarity | ~0.65 |
| Mean diff-speaker similarity | ~0.10 |

The 10x EER degradation vs. same-domain (~0.8%) is expected and typical for cross-domain (Western VoxCeleb → Indian-accent Lahaja) evaluation.

---

### Task A.3 — Gender Separability in Embedding Space

Applied **t-SNE** and **UMAP** (cosine metric) to 192-dim embeddings, colored by gender.

| Method | Observation |
|--------|-------------|
| t-SNE | Two broad regions — Male left, Female right — with some central overlap |
| UMAP (cosine) | **Perfect separation**, zero overlap, individual speaker sub-clusters visible |

**PCA analysis** — how many dimensions encode gender?

| PCA Components | Gender Accuracy |
|----------------|-----------------|
| **1** | **99.2%** |
| 3 | 99.6% |
| 20 | 100.0% |
| 192 | 100.0% |

**Interpretation**: Gender is concentrated in the **single most dominant axis** of the embedding space. Speaker identity requires ~125 dimensions to reach 95% explained variance — it is distributed, not concentrated. This is emergent disentanglement from learning speaker identity alone.

---

### Task A.4 — Adapter-Based Gender Classifier

Frozen ECAPA-TDNN backbone + lightweight classifier head. Two models evaluated:

| Model | Accuracy | F1 | CV Mean (5-fold) |
|-------|----------|----|-----------------|
| Logistic Regression | **100.0%** | 1.0000 | **99.62% ± 0.19%** |
| SVM (RBF kernel) | **100.0%** | 1.0000 | **99.81% ± 0.23%** |
| Random Baseline | 50.0% | 0.5000 | — |

Both LR and SVM achieved perfect accuracy, confirming **gender is linearly separable** in the embedding space — the RBF kernel was not needed. The near-zero overfit gap (0.0038 for LR) across all 5 folds confirms this is robust, not a lucky split.

---

### Bonus — Gradio Interactive Demo

A fully functional web UI wrapping the complete inference pipeline.

**Features:**
- Audio input via **file upload** or **microphone recording**
- Real-time **gender prediction** with confidence % for each speaker
- **Cosine similarity score** between two audio clips
- Adjustable **verification threshold slider** (default: EER threshold)
- Visual **similarity gauge** with color gradient
- **Public shareable URL** via Gradio tunneling

```bash
cd demo/
python app.py
# → Running on http://localhost:7860
# → Public URL: https://xxxxx.gradio.live
```

> **Note on microphone audio**: Studio-calibrated EER threshold (0.30) may need lowering to 0.10–0.15 for real-world mic recordings due to domain mismatch. The threshold slider makes this transparent and tunable.

---

## ⚙️ Setup & Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:**
```
speechbrain>=0.5.15
torch>=2.0.0
torchaudio>=2.0.0
torchcodec
datasets>=2.14.0          # HuggingFace — for Lahaja
scikit-learn>=1.3.0
umap-learn>=0.5.3
gradio>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
```

### Run the Full Pipeline

```bash
# 1. Download dataset and construct trial pairs (Task A.1)
python src/dataset_construction.py

# 2. Extract embeddings + run verification (Task A.2)
python src/embedding_extraction.py

# 3. Visualize gender separability (Task A.3)
python src/gender_separability.py

# 4. Train gender classifier + cross-validation (Task A.4)
python src/gender_classifier.py

# 5. Launch the Gradio demo
python demo/app.py
```

Or run everything end-to-end in the notebook:
```bash
jupyter notebook notebooks/speaker_verification_full.ipynb
```

---

## 📊 Dataset

**[ai4bharat/Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja)**

| Property | Value |
|----------|-------|
| Total duration | 12.5 hours |
| Speakers | 132 (65 Male, 67 Female) |
| Language | Hindi (multi-accent) |
| Geographic coverage | 83 districts across India |
| Avg utterances/speaker | ~8 |

---

## 🤖 Model

**[speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)**

| Property | Value |
|----------|-------|
| Architecture | ECAPA-TDNN (SE blocks, multi-scale aggregation) |
| Training data | VoxCeleb1+2 (1M+ utterances, 7000+ speakers) |
| Output | 192-dimensional speaker embedding |
| Input | 16kHz mono waveform |
| Same-domain EER | ~0.8% (VoxCeleb test) |

---

## 📈 Results Summary

### Speaker Verification — Threshold Sweep

| Threshold | TAR | FAR | Accuracy |
|-----------|-----|-----|----------|
| 0.10 | 0.987 | 0.590 | 69.8% |
| 0.20 | 0.950 | 0.213 | 86.8% |
| **0.30 (EER)** | **0.910** | **0.073** | **91.8%** |
| 0.40 | 0.807 | 0.007 | 90.0% |
| 0.50 | 0.703 | 0.000 | 85.2% |

### Gender Classification — 5-Fold CV

| Fold | LR Test | SVM Test |
|------|---------|----------|
| 1 | 99.53% | 100.00% |
| 2 | 99.53% | 99.53% |
| 3 | 99.53% | 99.53% |
| 4 | 100.00% | 100.00% |
| 5 | 99.53% | 100.00% |
| **Mean** | **99.62% ± 0.19%** | **99.81% ± 0.23%** |

---

## 🔍 Key Takeaways

1. **ECAPA-TDNN generalizes to Indian-accented speech** without any fine-tuning — EER 8.2% vs. ~50% random baseline.
2. **Gender emerges as the dominant axis** in a model trained purely on speaker identity — 1 PCA dimension achieves 99.2% gender accuracy.
3. **Linear separability**: Logistic regression on frozen 192-dim embeddings achieves perfect gender classification — no fine-tuning of the base model needed.
4. **Bias-aware evaluation matters**: Gender-balanced negative pairs are critical for honest EER reporting.
5. **Real-world deployment** requires threshold recalibration — studio EER thresholds don't directly transfer to microphone audio.

---

## 📄 Report

Full technical report (15 pages) available in [`docs/Speaker_Verification_Report.pdf`](docs/Speaker_Verification_Report.pdf).

Covers:
- Detailed sampling strategy justification and bias analysis
- Embedding health check and extraction pipeline
- PCA analysis with component-by-component accuracy breakdown
- Cross-validation methodology and overfitting discussion
- Gradio demo architecture and domain-mismatch findings

---

## 🙏 Acknowledgements

- [SpeechBrain](https://speechbrain.github.io/) — pre-trained ECAPA-TDNN model
- [AI4Bharat](https://ai4bharat.org/) — Lahaja dataset
- [HuggingFace Datasets](https://huggingface.co/datasets) — dataset loading infrastructure

---

## 📬 Contact

Built as part of a technical assessment for **Arrowhead**.  
Feel free to reach out with questions or collaboration ideas.
