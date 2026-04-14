import pandas as pd
import numpy as np
import torch
import h5py
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

def get_sequence_window(sequence, center_pos, window_size=201):
    half_window = window_size // 2
    start = max(0, center_pos - half_window)
    end = min(len(sequence), center_pos + half_window + 1)
    
    window = sequence[start:end]
    
    pad_left = max(0, half_window - center_pos)
    pad_right = max(0, (center_pos + half_window + 1) - len(sequence))
    
    if pad_left > 0:
        window = 'N' * pad_left + window
    if pad_right > 0:
        window = window + 'N' * pad_right
        
    return window

def extract_features(csv_path, output_h5_path, model_path="/zhouting/lucavirus", seq_len=201, batch_size=128):
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    
    required_cols = ['context', 'rnastructure']
    if 'edit_type' in data.columns:
        required_cols.append('edit_type')
    data = data.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Valid data rows: {len(data)}")
    
    if len(data) == 0:
        print("No valid data found. Skipping.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading LucaVirus model from {model_path} to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    total_samples = len(data)
    
    with h5py.File(output_h5_path, 'w') as f:
        feature_dataset = f.create_dataset(
            'features', 
            shape=(total_samples, seq_len, 2560), 
            dtype=np.float16,
            chunks=(1, seq_len, 2560)
        )
        
        for i in tqdm(range(0, total_samples, batch_size), desc=f"Extracting {os.path.basename(csv_path)}"):
            batch_data = data.iloc[i:i+batch_size]
            windows = []
            for _, row in batch_data.iterrows():
                seq = row['context']
                cp = len(seq) // 2
                window = get_sequence_window(seq, cp, seq_len)
                windows.append(window)
                
            fixed_max_len = seq_len + 2
            inputs = tokenizer(
                windows,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=fixed_max_len,
                add_special_tokens=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items() if k != 'token_type_ids'}
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs)
                    hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                    embs = hidden[:, 1:-1, :].cpu().numpy().astype(np.float16)
                    
            actual_batch_size = embs.shape[0]
            feature_dataset[i:i+actual_batch_size] = embs

if __name__ == "__main__":
    splits = ['train', 'val', 'test']
    for split in splits:
        csv_file = f"/zhouting/virus splits/{split}.csv"
        h5_file = f"/zhouting/virus splits/{split}_features.h5"
        if os.path.exists(csv_file):
            extract_features(csv_file, h5_file)
            print(f"Successfully created {h5_file}")
