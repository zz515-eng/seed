import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append('/zhouting')
import model as model_module
from experiment_v3_regularization.finetune_v3 import VirusRnaDataset

def evaluate_on_test():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    # Load test dataset
    print("Loading old test dataset (avoid50/N=248)...")
    test_dataset = VirusRnaDataset("/zhouting/virus splits/test.csv", "/zhouting/virus splits/test_features.h5", tokenizer=None, lucavirus_model=None, device=DEVICE, seq_len=201, positive_class="A-to-G")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    print("Loading the new model trained on sars2_ultimate (N=637)...")
    model = model_module.RnaClassifier(embedding_dim=2560, trinuc_vocab_size=65, trinuc_embedding_dim=2560, sec_struct_vocab_size=3, sec_struct_emb_dim=2560, num_classes=2).to(DEVICE)
    model.embedding_dim = 1024
    
    # Needs to match the Dropout architecture of V3
    model.conformer.dropout = torch.nn.Dropout(0.3)
    for layer in model.conformer.layers:
        layer.ff1.dropout1 = torch.nn.Dropout(0.3)
        layer.ff1.dropout2 = torch.nn.Dropout(0.3)
        layer.ff2.dropout1 = torch.nn.Dropout(0.3)
        layer.ff2.dropout2 = torch.nn.Dropout(0.3)
        layer.attn.dropout = torch.nn.Dropout(0.3)
        layer.conv.dropout = torch.nn.Dropout(0.3)
    model.classifier_head[2] = torch.nn.Dropout(0.5)

    output_dir = "/zhouting/viral_dataset_expansion/massive_run/new_model_on_old_data_results"
    os.makedirs(output_dir, exist_ok=True)
    model.load_state_dict(torch.load("/zhouting/viral_dataset_expansion/massive_run/v3_sars2_results/best_v3_sars2_model.pt", map_location=DEVICE))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    print("Running evaluation on the independent test set...")
    with torch.no_grad():
        for batch in test_loader:
            seq_embeddings = batch['seq_embeddings'].to(DEVICE)
            trinuc_indices = batch['trinuc_indices'].to(DEVICE)
            sec_struct_indices = batch['sec_struct_indices'].to(DEVICE)
            shape_values = batch['shape_values'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            with torch.amp.autocast('cuda'):
                logits = model(seq_embeddings, trinuc_indices, sec_struct_indices, shape_values)
                
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print("\n" + "="*50)
    print("🚀 FINAL TEST SET PERFORMANCE (V3 STRATEGY) 🚀")
    print("="*50)
    print(f"Total Test Samples: {len(all_labels)}")
    print(f"Confusion Matrix: \n{cm}")
    print(f"  - True Negatives (Correct Non-edited): {tn}")
    print(f"  - False Positives (False Alarms): {fp}")
    print(f"  - False Negatives (Missed Mutations): {fn}")
    print(f"  - True Positives (Correct Mutations): {tp}")
    print("-" * 50)
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"ROC-AUC Score:     {auc:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print("-" * 50)
    print(f"Metrics specifically for 'A-to-G' (Positive Class):")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print("="*50)

    # 1. Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Non-edited (0)', 'A-to-G (1)'], 
                yticklabels=['Non-edited (0)', 'A-to-G (1)'],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title(f'V3 Test Confusion Matrix\nAccuracy: {acc:.2%}', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v3_test_cm.png'), dpi=300)
    plt.close()

    # 2. Save ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('V3 Strategy Test ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v3_test_roc.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    evaluate_on_test()