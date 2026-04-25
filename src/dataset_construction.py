"""
Task A.1 — Trial Pair Generation for Speaker Verification
==========================================================
Constructs a gender-balanced verification trial set from the ai4bharat/Lahaja dataset.

Strategy:
- 300 positive pairs (same speaker, different utterances)
- 300 negative pairs: 150 same-gender + 150 cross-gender
- Seed = 42 for full reproducibility
- Canonical key deduplication: (min(idx_a, idx_b), max(idx_a, idx_b))

Why gender-balanced negatives?
  If all negatives are cross-gender, the model can score high by detecting gender
  differences alone — not actual speaker identity. Balancing forces the model to
  discriminate between same-gender speakers, which is the meaningful evaluation.
"""

import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from itertools import combinations

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_POSITIVE = 300
N_NEGATIVE = 300  # 150 same-gender + 150 cross-gender


def load_lahaja():
    """Load the Lahaja dataset from HuggingFace."""
    print("Loading ai4bharat/Lahaja ...")
    dataset = load_dataset("ai4bharat/Lahaja", split="test", trust_remote_code=True)
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} utterances from {df['speaker_id'].nunique()} speakers")
    return df


def build_speaker_index(df):
    """Build per-speaker utterance lists and gender lookup."""
    speaker_utts = df.groupby("speaker_id").apply(
        lambda g: list(g.index)
    ).to_dict()

    # Gender lookup: speaker_id -> 'M' or 'F'
    gender_map = df.drop_duplicates("speaker_id").set_index("speaker_id")["gender"].to_dict()

    # Filter: only speakers with >= 2 utterances can form positive pairs
    eligible = {spk: idxs for spk, idxs in speaker_utts.items() if len(idxs) >= 2}
    print(f"Eligible speakers (≥2 utterances): {len(eligible)} / {len(speaker_utts)}")
    return eligible, gender_map


def generate_positive_pairs(speaker_utts, n=N_POSITIVE):
    """Sample n positive pairs (same speaker, different utterances)."""
    pairs = []
    seen = set()
    speakers = list(speaker_utts.keys())

    attempts = 0
    while len(pairs) < n and attempts < n * 100:
        attempts += 1
        spk = random.choice(speakers)
        utts = speaker_utts[spk]
        a, b = random.sample(utts, 2)
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            pairs.append({"idx_a": a, "idx_b": b, "label": 1, "speaker_a": spk, "speaker_b": spk})

    print(f"Generated {len(pairs)} positive pairs")
    return pairs, seen


def generate_negative_pairs(speaker_utts, gender_map, seen, n_same=150, n_cross=150):
    """
    Sample negative pairs with explicit gender balancing.

    n_same:  same-gender negatives (M-M or F-F)
    n_cross: cross-gender negatives (M-F)

    Gender balance prevents model from exploiting gender cues for easy rejections.
    """
    speakers = list(speaker_utts.keys())
    male_spks   = [s for s in speakers if gender_map.get(s) == "M"]
    female_spks = [s for s in speakers if gender_map.get(s) == "F"]

    def sample_pair(pool_a, pool_b, label_type):
        a_spk = random.choice(pool_a)
        b_spk = random.choice(pool_b)
        while b_spk == a_spk:
            b_spk = random.choice(pool_b)
        a_idx = random.choice(speaker_utts[a_spk])
        b_idx = random.choice(speaker_utts[b_spk])
        return a_idx, b_idx, a_spk, b_spk

    pairs = []

    # --- Same-gender negatives ---
    attempts = 0
    while len(pairs) < n_same and attempts < n_same * 200:
        attempts += 1
        pool = male_spks if random.random() < 0.5 else female_spks
        a_idx, b_idx, a_spk, b_spk = sample_pair(pool, pool, "same_gender")
        key = (min(a_idx, b_idx), max(a_idx, b_idx))
        if key not in seen:
            seen.add(key)
            pairs.append({
                "idx_a": a_idx, "idx_b": b_idx,
                "label": 0,
                "speaker_a": a_spk, "speaker_b": b_spk,
                "neg_type": "same_gender"
            })

    # --- Cross-gender negatives ---
    attempts = 0
    cross_pairs = []
    while len(cross_pairs) < n_cross and attempts < n_cross * 200:
        attempts += 1
        a_idx, b_idx, a_spk, b_spk = sample_pair(male_spks, female_spks, "cross_gender")
        key = (min(a_idx, b_idx), max(a_idx, b_idx))
        if key not in seen:
            seen.add(key)
            cross_pairs.append({
                "idx_a": a_idx, "idx_b": b_idx,
                "label": 0,
                "speaker_a": a_spk, "speaker_b": b_spk,
                "neg_type": "cross_gender"
            })

    all_neg = pairs + cross_pairs
    print(f"Generated {len(pairs)} same-gender + {len(cross_pairs)} cross-gender = {len(all_neg)} negative pairs")
    return all_neg


def build_trial_set(df):
    """Full pipeline: load → index → sample → save."""
    speaker_utts, gender_map = build_speaker_index(df)

    pos_pairs, seen = generate_positive_pairs(speaker_utts)
    neg_pairs = generate_negative_pairs(speaker_utts, gender_map, seen)

    all_pairs = pos_pairs + neg_pairs
    trial_df = pd.DataFrame(all_pairs)
    trial_df = trial_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nFinal trial set: {len(trial_df)} pairs")
    print(trial_df["label"].value_counts().rename({1: "positive", 0: "negative"}))

    trial_df.to_csv("results/metrics/trial_pairs.csv", index=False)
    print("Saved → results/metrics/trial_pairs.csv")
    return trial_df


if __name__ == "__main__":
    df = load_lahaja()
    trial_df = build_trial_set(df)
    print(trial_df.head(10))
