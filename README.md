# Project Seed: Model Usage and Data Pipeline

This repository contains the core pipeline for Project Seed, including data preparation, model architecture, inference scripts, and pre-trained weights.

## 1. Directory Structure

Current model, October 2026. FiLM, center-base readout, dropout 0.1.

- `model/model.py`: `RnaClassifier`.
- `model/dataset_loader.py`: `VirusRnaDataset`. Reads a CSV and an H5 in the same row order. Class 0 is edited.
- `weights/base_human_model.pt`: Human ALU pretraining checkpoint. This is the starting point for the virus runs.
- `weights/finetune/`: Leave-one-virus fine-tunes on `data/sites.csv`. One file each for HCV, MV, NoVGII, and HHV8. `results.txt` is the score record for that run.
- `data/sites.csv`: 2958 rows, six viruses, edited and not-edited counts matched. Columns include `context`, `rnastructure`, and `edit_type`.
- `inference/example_inference.py`: Loads the FiLM model. Default weights are `weights/base_human_model.pt`. Default table is `data/sites.csv`.
- `data_prep/extract_virus_features.py`: Builds the H5 from a CSV. The 2958-row feature file is about 3 GB and is not in this repository.
- `archive/20260414_concat_n160/`: The April concatenated model and the N=160 avoid50 tables. Those weights do not load into `model.py`.

## 2. Data Preparation for Custom Data

If you want to use this model to evaluate your own viral RNA sequences, you must prepare your data in a specific format and extract the deep learning features.

1. **Dataset Format**:
   - Prepare your sequence data in a `.csv` format. You can refer to `data/sites.csv` for an example.
   - Your `.csv` file MUST contain the following columns:
     - `context`: The RNA/DNA sequence context around the candidate editing site. The sequence should be exactly **201 nucleotides** long, with the candidate mutation site at the center (position 101).
     - `rnastructure`: The secondary structure of the `context` sequence in dot-bracket notation (length 201). **You must calculate this using a tool like [ViennaRNA RNAfold](https://www.tbi.univie.ac.at/RNA/RNAfold.1.html)** before passing the data to our pipeline.
     - `edit_type`: `A-to-G` or `Edited` for an edited site, `Non-edited` otherwise. The network uses class 0 for edited and class 1 for not edited.
     - `label`: Optional, and not present in `data/sites.csv`. The loader uses `edit_type`.
2. **Feature Extraction**:
   - Once your `.csv` is ready with `context` and `rnastructure`, use `data_prep/extract_virus_features.py` to convert the `.csv` sequences into a `.h5` feature matrix using the foundation model.
   - *Note: Due to file size limits, the large `.h5` feature matrices are not included in this repository and must be generated locally.*

## 3. How to Use the Model (Inference)

To load the model and perform inference, you must instantiate the architecture with the specific hyperparameters used during training.

### Example Code:

```python
import torch
from torch.utils.data import DataLoader

# Import custom modules
from model.model import RnaClassifier
from model.dataset_loader import VirusRnaDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Dataset
dataset = VirusRnaDataset(
    csv_path="data/sites.csv",
    h5_path="path/to/your/generated_features.h5",
    tokenizer=None, lucavirus_model=None, device=DEVICE, seq_len=201, positive_class="A-to-G"
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 2. Initialize the FiLM model. Dropout stays 0.1. Read the center base.
model = RnaClassifier(
    embedding_dim=2560, trinuc_vocab_size=65, trinuc_embedding_dim=2560,
    sec_struct_vocab_size=3, sec_struct_emb_dim=2560, num_classes=2,
    dropout=0.1, attention_dropout=0.1, pooling_strategy="center",
).to(DEVICE)

# 3. Load the human starting checkpoint, or one file under weights/finetune/.
model.load_state_dict(torch.load("weights/base_human_model.pt", map_location=DEVICE))
model.eval()

# 4. Inference Loop
with torch.no_grad():
    for batch in dataloader:
        # Pass batch to model
        # ...
```

For a complete working example, refer to `inference/example_inference.py`.

## 4. Results Interpretation

The model performs binary classification to predict RNA editing events:
- **Class 0**: Edited site (A-to-G).
- **Class 1**: Not edited.

**Expected Output**:
The model outputs raw logits. After softmax, a higher probability for class 0 means the center base is predicted as edited.

The network uses FiLM and reads the center position. Leave dropout at 0.1. Do not replace dropout layers after the model is created.
