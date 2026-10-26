"""Load a CSV and its LucaVirus H5 for the FiLM model.

Class 0 is the edited base. Class 1 is not edited. Dropout is not changed here.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN = 201
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


class VirusRnaDataset(Dataset):
    """Rows stay aligned with the H5. Missing fields are not dropped."""

    def __init__(self, csv_path, h5_path, tokenizer=None, lucavirus_model=None, device=None, seq_len=SEQ_LEN, positive_class="A-to-G"):
        del tokenizer, lucavirus_model, device
        if positive_class not in EDITED:
            raise ValueError(f"positive_class must be an edited label, got {positive_class!r}")
        if seq_len != SEQ_LEN:
            raise ValueError(f"This model uses a window of {SEQ_LEN}, got {seq_len}")

        frame = pd.read_csv(csv_path)
        required = ["context", "rnastructure", "edit_type"]
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        if frame[required].isna().any().any():
            raise ValueError("CSV rows are missing context, structure, or edit_type. Dropping them would break H5 alignment.")

        self.contexts = frame["context"].astype(str).tolist()
        self.structures = frame["rnastructure"].astype(str).tolist()
        self.labels = np.array([class_id(value) for value in frame["edit_type"]], dtype=np.int64)
        self.seq_len = seq_len

        import h5py

        self._handle = h5py.File(h5_path, "r")
        self.features = self._handle["features"]
        if tuple(self.features.shape[1:]) != (SEQ_LEN, 2560) or int(self.features.shape[0]) != len(frame):
            raise ValueError(f"H5 shape {self.features.shape} does not match {len(frame)} CSV rows of length {SEQ_LEN}.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        context = self.contexts[index]
        structure = self.structures[index]
        center = len(context) // 2
        seq = sequence_window(context, center, self.seq_len)
        dots = sequence_window(structure, center, self.seq_len)
        embedding = torch.from_numpy(np.asarray(self.features[index], dtype=np.float16))
        return {
            "seq_embeddings": embedding,
            "trinuc_indices": torch.tensor(trinuc_indices(seq), dtype=torch.long),
            "sec_struct_indices": torch.tensor(structure_indices(dots), dtype=torch.long),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }
