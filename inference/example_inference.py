"""Score a CSV with the FiLM model.

Dropout stays 0.1. The readout is the center base. Do not raise dropout
after loading. The weights currently in weights/ are the older concatenated
model and will not load into this network.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model import RnaClassifier  # noqa: E402

SEQ_LEN = 201
# Class 0 is the edited base. Class 1 is not edited.
EDITED = {"A-to-G", "Edited"}
NOT_EDITED = {"Non-edited"}


def build_trinucleotide_map():
    bases = ["A", "C", "G", "T"]
    mapping = {
        b1 + b2 + b3: i
        for i, (b1, b2, b3) in enumerate((x, y, z) for x in bases for y in bases for z in bases)
    }
    for b1 in bases + ["N"]:
        for b2 in bases + ["N"]:
            for b3 in bases + ["N"]:
                token = b1 + b2 + b3
                if token not in mapping:
                    mapping[token] = 0
    return mapping


TRINUC_MAP = build_trinucleotide_map()


def sequence_window(sequence, center, window_size):
    start = max(0, center - window_size // 2)
    end = start + window_size
    window = sequence[start:end]
    if len(window) < window_size:
        if start == 0 and center < window_size // 2:
            window = "N" * (window_size // 2 - center) + window
        if end >= len(sequence):
            window = window + "N" * (window_size - len(window))
    return window[:window_size].ljust(window_size, "N")


def trinuc_indices(window):
    dna = window.upper().replace("U", "T")
    padded = "N" + dna + "N"
    return [TRINUC_MAP.get(padded[i : i + 3], 0) for i in range(len(window))]


def structure_indices(window):
    mapping = {".": 0, "(": 1, ")": 2}
    return [mapping.get(char, 0) for char in window]


def class_id(edit_type):
    name = str(edit_type)
    if name in EDITED:
        return 0
    if name in NOT_EDITED:
        return 1
    raise ValueError(f"Unexpected edit_type {name!r}. Use A-to-G or Non-edited.")


class CsvH5Dataset(Dataset):
    def __init__(self, csv_path, h5_path):
        frame = pd.read_csv(csv_path)
        required = ["context", "rnastructure", "edit_type"]
        before = len(frame)
        frame = frame.dropna(subset=required).reset_index(drop=True)
        if len(frame) != before:
            raise ValueError("CSV rows are missing context, structure, or edit_type. Dropping them would break H5 alignment.")
        self.contexts = frame["context"].astype(str).tolist()
        self.structures = frame["rnastructure"].astype(str).tolist()
        self.labels = np.array([class_id(v) for v in frame["edit_type"]], dtype=np.int64)
        import h5py

        handle = h5py.File(h5_path, "r")
        self.features = handle["features"]
        self._handle = handle
        if tuple(self.features.shape[1:]) != (SEQ_LEN, 2560) or int(self.features.shape[0]) != len(frame):
            raise ValueError(f"H5 shape {self.features.shape} does not match {len(frame)} CSV rows of length {SEQ_LEN}.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        context = self.contexts[index]
        structure = self.structures[index]
        center = len(context) // 2
        seq = sequence_window(context, center, SEQ_LEN)
        dots = sequence_window(structure, center, SEQ_LEN)
        embedding = torch.from_numpy(np.asarray(self.features[index], dtype=np.float16))
        return {
            "seq_embeddings": embedding,
            "trinuc_indices": torch.tensor(trinuc_indices(seq), dtype=torch.long),
            "sec_struct_indices": torch.tensor(structure_indices(dots), dtype=torch.long),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


def load_weights(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise SystemExit(f"{path} is not a state dict.")
    old = any(key.startswith("feature_projection") for key in state)
    film = any(key.startswith("film_gen") for key in state)
    if old and not film:
        raise SystemExit(
            f"{path} is the older concatenated model. "
            "This script loads the FiLM model and reads the center base. "
            "The matching weights are not in this repository yet."
        )
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "sites.csv")
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "base_human_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RnaClassifier(pooling_strategy="center", dropout=0.1, attention_dropout=0.1).to(device)
    model.load_state_dict(load_weights(args.weights, device))
    model.eval()

    loader = DataLoader(CsvH5Dataset(args.csv, args.h5), batch_size=args.batch_size, shuffle=False)
    labels, edited_prob, predictions = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["seq_embeddings"].to(device),
                batch["trinuc_indices"].to(device),
                batch["sec_struct_indices"].to(device),
            )
            probability = torch.softmax(logits, dim=1)[:, 0]
            edited_prob.extend(probability.cpu().numpy())
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch["label"].numpy())

    labels = np.asarray(labels)
    edited_prob = np.asarray(edited_prob)
    predictions = np.asarray(predictions)
    print(f"rows {len(labels)}")
    print(f"accuracy {accuracy_score(labels, predictions):.4f}")
    print(f"MCC {matthews_corrcoef(labels, predictions):.4f}")
    # Score class 0, the edited class, against the numeric labels.
    print(f"AUC {roc_auc_score(labels == 0, edited_prob):.4f}")


if __name__ == "__main__":
    main()
