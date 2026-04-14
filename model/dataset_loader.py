import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from tqdm import tqdm

sys.path.append('/zhouting')
import model as model_module
from train import RnaDataset

class VirusRnaDataset(RnaDataset):
    def __init__(self, csv_path, h5_path, *args, **kwargs):
        super().__init__(csv_path, *args, **kwargs)
        self.h5_path = h5_path

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            seq_embeddings = batch['seq_embeddings'].to(device)
            trinuc_indices = batch['trinuc_indices'].to(device)
            sec_struct_indices = batch['sec_struct_indices'].to(device)
            shape_values = batch['shape_values'].to(device)
            labels = batch['label'].to(device)

            with torch.amp.autocast('cuda'):
                logits = model(seq_embeddings, trinuc_indices, sec_struct_indices, shape_values)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    try: metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
    except: metrics['roc_auc'] = 0.5
    return metrics

def run_finetune():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "/zhouting/experiment_v3_regularization/results"
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    train_dataset = VirusRnaDataset("/zhouting/virus splits/train.csv", "/zhouting/virus splits/train_features.h5", tokenizer=None, lucavirus_model=None, device=DEVICE, seq_len=201, positive_class="A-to-G")
    val_dataset = VirusRnaDataset("/zhouting/virus splits/val.csv", "/zhouting/virus splits/val_features.h5", tokenizer=None, lucavirus_model=None, device=DEVICE, seq_len=201, positive_class="A-to-G")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize Model
    model = model_module.RnaClassifier(embedding_dim=2560, trinuc_vocab_size=65, trinuc_embedding_dim=2560, sec_struct_vocab_size=3, sec_struct_emb_dim=2560, num_classes=2).to(DEVICE)
    model.embedding_dim = 1024
    
    # ADD STRONGER DROPOUT for regularization
    model.conformer.dropout = nn.Dropout(0.3) # Increased from 0.1
    for layer in model.conformer.layers:
        layer.ff1.dropout1 = nn.Dropout(0.3)
        layer.ff1.dropout2 = nn.Dropout(0.3)
        layer.ff2.dropout1 = nn.Dropout(0.3)
        layer.ff2.dropout2 = nn.Dropout(0.3)
        layer.attn.dropout = nn.Dropout(0.3)
        layer.conv.dropout = nn.Dropout(0.3)
    
    model.classifier_head[2] = nn.Dropout(0.5) # Increased from 0.1
    
    model.load_state_dict(torch.load("/zhouting/training_results_full/conformer/ce/best_model.pt", map_location=DEVICE), strict=False)

    print("\n🚀 Starting V3: DLR + Label Smoothing + High Weight Decay + High Dropout 🚀")
    
    # STRATEGY V3: DLR + Stronger Regularization
    base_params, conformer_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if "classifier_head" in name:
            head_params.append(param)
        elif "conformer" in name:
            conformer_params.append(param)
        else:
            base_params.append(param)

    # Increased weight decay significantly (from 1e-3 to 1e-2) to penalize large weights
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 1e-6},
        {'params': conformer_params, 'lr': 5e-5},
        {'params': head_params, 'lr': 5e-4}
    ], weight_decay=1e-2) 

    # Implement Label Smoothing (0.1 means target becomes 0.9 and 0.1 instead of 1.0 and 0.0)
    # This prevents the model from becoming overconfident on the small training set
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    history = []

    for epoch in range(1, 21):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(batch['seq_embeddings'].to(DEVICE), batch['trinuc_indices'].to(DEVICE), batch['sec_struct_indices'].to(DEVICE), batch['shape_values'].to(DEVICE))
                loss = criterion(logits, batch['label'].to(DEVICE))
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        scheduler.step()
            
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")
        
        history.append({'epoch': epoch, 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['accuracy']})
        
        # Save based on Accuracy instead of Loss because Label Smoothing changes the loss scale
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_v3_model.pt'))

    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'history.csv'), index=False)
    print(f"Done! Saved to {output_dir}")

if __name__ == "__main__":
    run_finetune()
