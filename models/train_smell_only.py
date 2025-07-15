import os, torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn, torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics       import precision_score, recall_score, f1_score, roc_auc_score

from torch.utils.tensorboard import SummaryWriter
import wandb

from models.smell_only_dataset  import SmellOnlyDataset
from models.odor_net_smell_only import OdorNet

def train():
    # 1) Config
    SENSOR_ROOT, DESC_PATH = "smell_data", "smell_descriptions.json"
    BATCH_SIZE, EPOCHS, LR = 32, 20, 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Dataset + split
    full_ds = SmellOnlyDataset(SENSOR_ROOT, DESC_PATH, sensor_seq_len=600)
    idxs   = list(range(len(full_ds)))
    labels = [full_ds[i]['label'] for i in idxs]
    train_idx, val_idx = train_test_split(
        idxs, test_size=0.2, stratify=labels, random_state=42
    )
    train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3) Model, loss, optimizer
    model     = OdorNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4) Init logging
    writer = SummaryWriter("runs/exp1")
    wandb.init(
      project="smell-classifier",
      name="run1",
      config={"batch_size":BATCH_SIZE, "lr":LR, "epochs":EPOCHS}
    )
    wandb.watch(model, log="all", log_freq=10)

    best_f1, no_improve = 0.0, 0
    PATIENCE = 5

    # 5) Loop epoche
    for epoch in range(1, EPOCHS+1):
        # — TRAIN —
        model.train()
        tot_loss = 0
        for b in train_loader:
            x, y = b['sensor'].to(DEVICE), b['label'].to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * x.size(0)
        avg_train_loss = tot_loss / len(train_ds)

        # — VALIDATION —
        model.eval()
        val_loss = 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for b in val_loader:
                x, y = b['sensor'].to(DEVICE), b['label'].to(DEVICE)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                probs  = torch.softmax(logits, dim=1)
                preds  = logits.argmax(dim=1)
                all_probs .extend(probs.cpu().numpy())
                all_preds .extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        avg_val_loss = val_loss / len(val_ds)

        # — METRICHE —
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = float('nan')

        # — CHECKPOINT & EARLY STOPPING —
        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_odornet.pt")
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Stop early at epoch {epoch}")
            break

        # — LOG su console —
        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")
        print(f" → Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")

        # — LOG su TensorBoard & W&B —
        writer.add_scalars("Loss", {"train":avg_train_loss,"val":avg_val_loss}, epoch)
        writer.add_scalars("F1",   {"val":f1}, epoch)
        wandb.log({
          "epoch": epoch,
          "train_loss": avg_train_loss,
          "val_loss":   avg_val_loss,
          "precision":  precision,
          "recall":     recall,
          "f1":         f1,
          "roc_auc":    roc_auc
        })

    writer.close()
    print("Training completato. Best F1:", best_f1)

if __name__ == "__main__":
    train()
