"""
Task A.2 — ECAPA-TDNN Embedding Extraction & Speaker Verification
==================================================================
Extracts 192-dim speaker embeddings using the pre-trained ECAPA-TDNN model
from SpeechBrain, then evaluates speaker verification on the trial pairs.

Key design: unique-index caching reduces computation from O(pairs) → O(unique utterances).

Model: speechbrain/spkrec-ecapa-voxceleb
Input:  16kHz mono waveform
Output: 192-dimensional L2-normalized speaker embedding
"""

import numpy as np
import pandas as pd
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from pathlib import Path

SAMPLE_RATE = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

def load_model():
    """Load pre-trained ECAPA-TDNN from SpeechBrain."""
    print("Loading ECAPA-TDNN model...")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa-tdnn",
        run_opts={"device": DEVICE}
    )
    model.eval()
    print("Model loaded.")
    return model


# ---------------------------------------------------------------------------
# 2. Audio preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(audio_array, orig_sr):
    """
    Convert raw audio array to a 16kHz mono tensor.

    Steps:
      1. Convert numpy → torch float32
      2. If multi-channel: average to mono
      3. Resample to 16kHz
    """
    waveform = torch.tensor(audio_array, dtype=torch.float32)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)           # (1, T)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono average

    if orig_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform  # (1, T)


# ---------------------------------------------------------------------------
# 3. Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(dataset_df, model):
    """
    Extract 192-dim embeddings for all unique utterances.

    Uses unique-index caching: each utterance is processed exactly once,
    regardless of how many trial pairs reference it.

    Returns: dict {utterance_index: np.array(192,)}
    """
    trial_df = pd.read_csv("results/metrics/trial_pairs.csv")
    unique_indices = set(trial_df["idx_a"]) | set(trial_df["idx_b"])
    print(f"Extracting embeddings for {len(unique_indices)} unique utterances...")

    embeddings = {}

    for i, idx in enumerate(sorted(unique_indices)):
        row = dataset_df.iloc[idx]
        audio = row["audio"]
        waveform = load_and_preprocess(audio["array"], audio["sampling_rate"])
        waveform = waveform.to(DEVICE)

        with torch.no_grad():
            emb = model.encode_batch(waveform)   # (1, 1, 192)
            emb = emb.squeeze().cpu().numpy()    # (192,)

        embeddings[idx] = emb

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(unique_indices)} done")

    print(f"Extracted {len(embeddings)} embeddings. Shape: {embeddings[list(embeddings.keys())[0]].shape}")

    # Health check
    sample = list(embeddings.values())[0]
    print(f"\nEmbedding health check:")
    print(f"  Shape:      {sample.shape}")
    print(f"  L2 norm:    {np.linalg.norm(sample):.2f}")
    print(f"  Mean:       {sample.mean():.4f}")
    print(f"  Std:        {sample.std():.4f}")

    return embeddings


# ---------------------------------------------------------------------------
# 4. Cosine similarity + verification decisions
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def score_pairs(trial_df, embeddings):
    """Compute cosine similarity for every trial pair."""
    scores = []
    for _, row in trial_df.iterrows():
        emb_a = embeddings[row["idx_a"]]
        emb_b = embeddings[row["idx_b"]]
        scores.append(cosine_similarity(emb_a, emb_b))

    trial_df = trial_df.copy()
    trial_df["score"] = scores
    return trial_df


# ---------------------------------------------------------------------------
# 5. Evaluation metrics
# ---------------------------------------------------------------------------

def compute_eer(labels, scores):
    """Compute Equal Error Rate (EER) and its threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    return eer, threshold


def threshold_sweep(trial_df, thresholds=None):
    """Evaluate TAR / FAR / FRR / Accuracy at multiple thresholds."""
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    results = []
    for t in thresholds:
        decisions = (trial_df["score"] >= t).astype(int)
        pos_mask = trial_df["label"] == 1
        neg_mask = trial_df["label"] == 0

        tar = (decisions[pos_mask] == 1).mean()   # True Accept Rate
        far = (decisions[neg_mask] == 1).mean()   # False Accept Rate
        frr = 1 - tar                              # False Reject Rate
        acc = (decisions == trial_df["label"]).mean()

        results.append({"threshold": t, "TAR": tar, "FAR": far, "FRR": frr, "Accuracy": acc})

    return pd.DataFrame(results)


def run_verification(trial_df):
    """Full verification evaluation pipeline."""
    labels = trial_df["label"].values
    scores = trial_df["score"].values

    eer, eer_threshold = compute_eer(labels, scores)
    auc = roc_auc_score(labels, scores)

    print("\n" + "="*50)
    print("SPEAKER VERIFICATION RESULTS")
    print("="*50)
    print(f"EER:           {eer*100:.1f}%")
    print(f"EER Threshold: {eer_threshold:.2f}")
    print(f"AUC-ROC:       {auc:.4f}")

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    print(f"\nMean same-speaker score:  {pos_scores.mean():.4f}")
    print(f"Mean diff-speaker score:  {neg_scores.mean():.4f}")
    print(f"Separation gap:           {pos_scores.mean() - neg_scores.mean():.4f}")

    sweep_df = threshold_sweep(trial_df)
    print("\nThreshold sweep:")
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv("results/metrics/threshold_sweep.csv", index=False)

    plot_roc_curve(labels, scores, auc)
    plot_score_distribution(pos_scores, neg_scores, eer_threshold)

    return eer, auc, eer_threshold


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

def plot_roc_curve(labels, scores, auc):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#378ADD", lw=2, label=f"ECAPA-TDNN (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Speaker Verification on Lahaja")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/roc_curve.png", dpi=150)
    plt.close()
    print("Saved → results/plots/roc_curve.png")


def plot_score_distribution(pos_scores, neg_scores, eer_threshold):
    plt.figure(figsize=(8, 5))
    plt.hist(neg_scores, bins=40, alpha=0.6, color="#E24B4A", label="Different speakers (negative)")
    plt.hist(pos_scores, bins=40, alpha=0.6, color="#1D9E75", label="Same speaker (positive)")
    plt.axvline(eer_threshold, color="black", linestyle="--", lw=1.5, label=f"EER threshold ({eer_threshold:.2f})")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Count")
    plt.title("Score Distribution — Same vs Different Speakers")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/score_distribution.png", dpi=150)
    plt.close()
    print("Saved → results/plots/score_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datasets import load_dataset
    print("Loading Lahaja dataset...")
    dataset = load_dataset("ai4bharat/Lahaja", split="test", trust_remote_code=True)
    df = dataset.to_pandas()

    model = load_model()
    embeddings = extract_embeddings(df, model)

    # Save embeddings
    import pickle
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    with open("results/metrics/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    print("Saved → results/metrics/embeddings.pkl")

    trial_df = pd.read_csv("results/metrics/trial_pairs.csv")
    trial_df = score_pairs(trial_df, embeddings)
    trial_df.to_csv("results/metrics/trial_pairs_scored.csv", index=False)

    run_verification(trial_df)
