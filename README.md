# Project Seed: Model Usage and Data Pipeline

This repository contains the core pipeline for Project Seed, including data preparation, model architecture, inference scripts, and pre-trained weights.

## 1. Directory Structure
- `data_prep/`: Scripts for processing raw sequences and extracting features.
  - `extract_virus_features.py`: Python script to extract hidden representations (`.h5`) using a pre-trained foundation model.
- `model/`: Neural network architecture and dataset loader.
  - `model.py`: Core model definition (`RnaClassifier`).
  - `dataset_loader.py`: Contains the `VirusRnaDataset` class to load CSV labels and H5 features.
- `weights/`: Pre-trained model weights.
  - `best_seed_model.pt`: The fine-tuned weights for inference (Trained on the N=160 dataset).
- `data/`: Sample data.
  - `sample_train_ultimate.csv`: Example of the processed dataset format (N=160 samples).
- `inference/`: Example scripts for model evaluation.
  - `example_inference.py`: Demonstrates how to load the model, apply correct regularization, and evaluate on a dataset.

## 2. Data Preparation Pipeline

1. **Dataset Format**:
   - Prepare your sequence data in a `.csv` format. 
   - You can refer to `data/sample_train_ultimate.csv` for an example of the required columns (e.g., `context`, `label`, `rnastructure`).
2. **Feature Extraction**:
   - Use `data_prep/extract_virus_features.py` to convert the `.csv` sequences into a `.h5` feature matrix using the foundation model.
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
    csv_path="data/sample_train_ultimate.csv", 
    h5_path="path/to/your/generated_features.h5", 
    tokenizer=None, lucavirus_model=None, device=DEVICE, seq_len=201, positive_class="A-to-G"
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 2. Initialize Model Architecture
model = RnaClassifier(
    embedding_dim=2560, trinuc_vocab_size=65, trinuc_embedding_dim=2560, 
    sec_struct_vocab_size=3, sec_struct_emb_dim=2560, num_classes=2
).to(DEVICE)
model.embedding_dim = 1024

# 3. Apply Strong Regularization Settings (Required for this specific model)
model.conformer.dropout = torch.nn.Dropout(0.3)
for layer in model.conformer.layers:
    layer.ff1.dropout1 = torch.nn.Dropout(0.3)
    layer.ff1.dropout2 = torch.nn.Dropout(0.3)
    layer.ff2.dropout1 = torch.nn.Dropout(0.3)
    layer.ff2.dropout2 = torch.nn.Dropout(0.3)
    layer.attn.dropout = torch.nn.Dropout(0.3)
    layer.conv.dropout = torch.nn.Dropout(0.3)
model.classifier_head[2] = torch.nn.Dropout(0.5)

# 4. Load Weights
model.load_state_dict(torch.load("weights/best_seed_model.pt", map_location=DEVICE))
model.eval()

# 5. Inference Loop
with torch.no_grad():
    for batch in dataloader:
        # Pass batch to model
        # ...
```

For a complete working example, refer to `inference/example_inference.py`.

## 4. Results Interpretation

The model performs binary classification to predict RNA editing events:
- **Class 0 (Negative)**: Non-edited sequence.
- **Class 1 (Positive)**: Edited sequence (e.g., A-to-G mutation).

**Expected Output**:
The model outputs raw logits which can be converted to probabilities using a Softmax or Sigmoid function. A higher probability for Class 1 indicates a strong confidence that the central nucleotide in the given sequence context has undergone an editing event.

**Performance Note**:
This "Seed" model has been trained with strong regularization (Dropout 0.5, DLR) to prevent overfitting on specific sequence motifs, making it highly sensitive to true structural and contextual editing signals in viral RNA genomes.
