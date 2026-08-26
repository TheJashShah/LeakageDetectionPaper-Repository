 
import os

current_dir = os.getcwd()
Paper_dir = os.path.join(current_dir, "..")
log_dir = os.path.join(Paper_dir, "logs")
result_dir = os.path.join(Paper_dir, "results")
files_dir = os.path.join(Paper_dir, "files")

 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import random
import tqdm

 
from datetime import datetime
id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
n_scenarios = 100

 
WINDOW_SIZE = 12
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 10
DROPOUT = 0.5
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
log_file = open(os.path.join(log_dir, f"log_lstm_{id}.txt"), "a")
log_file.write(f"LSTM with {n_scenarios} scenarios.\n")
log_file.flush()

 
def load_scenario_data(scenario_id):

    data  = pd.read_csv(os.path.join(r"", f"Scenario-{scenario_id}.csv"))
    data = data.drop(columns=['Unnamed: 0'])

    return data

 
def add_temporal_columns(df):
    
    step_of_day = np.array([i % 48 for i in range(len(df))])
    
    df["sin_hour"] = np.sin(2 * np.pi * step_of_day / 48)
    df["cos_hour"] = np.cos(2 * np.pi * step_of_day / 48)
    
    return df

 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

 
class WindowedScenarioDatasetSeq(Dataset):
    
    def __init__(self, df_list, window_size=5, step=1):
        self.window_size = window_size
        self.step = step
        self.X = []
        self.y = []
        for df in df_list:
            arr = df.drop(columns=["Leaks"]).values.astype(np.float32)
            labs = df["Leaks"].values.astype(np.int64)
            T, Fdim = arr.shape
            for start in range(0, T - window_size + 1, step):
                window = arr[start:start + window_size]  # (window_size, n_features) - NOT flattened
                target = labs[start + window_size - 1]
                self.X.append(window)
                self.y.append(target)
        self.X = np.array(self.X, dtype=np.float32)  # (N, window_size, n_features)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

 
class LeakLSTM(nn.Module):

    def __init__(self, n_features, lstm_hidden=200, fcn1=100, fcn2=100, fcn3=50, dropout=DROPOUT):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.bn1 = nn.BatchNorm1d(lstm_hidden)

        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.fc_pre = nn.Sequential(
            nn.Linear(lstm_hidden, fcn1),
            nn.BatchNorm1d(fcn1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_block = nn.Sequential(
            nn.Linear(fcn1, fcn2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fcn2, fcn3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fcn3, 1)
        )

    def forward(self, x):
        # x: (batch, window_size, n_features)
        out, _ = self.lstm1(x)          # (batch, seq, lstm_hidden)
        last1 = out[:, -1, :]           # take last timestep -> (batch, lstm_hidden)
        last1 = self.bn1(last1)

        # feed last1 back through a single-step LSTM2 by unsqueezing to (batch, 1, hidden)
        out2, _ = self.lstm2(last1.unsqueeze(1))
        last2 = out2[:, -1, :]          # (batch, lstm_hidden)

        h = self.fc_pre(last2)
        logits = self.fc_block(h).squeeze(1)
        return torch.sigmoid(logits)

 
def compute_global_mean_std(df_list):
    sum_ = None
    sumsq_ = None
    n_total = 0
    for df in df_list:
        arr = df.drop(columns=["Leaks"]).values.astype(np.float64)
        if sum_ is None:
            sum_ = arr.sum(axis=0)
            sumsq_ = (arr**2).sum(axis=0)
        else:
            sum_ += arr.sum(axis=0)
            sumsq_ += (arr**2).sum(axis=0)
        n_total += arr.shape[0]
    mean = sum_ / n_total
    var = (sumsq_ / n_total) - (mean**2)
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_df(df, mean, std):
    cols = [c for c in df.columns if c != "Leaks"]
    df2 = df.copy()
    df2[cols] = (df2[cols] - mean) / std
    return df2

 
TRAIN_SCENARIOS = []
VAL_SCENARIOS = []
TEST_SCENARIOS = []

log_file.write(f"TRAIN_SCENARIOS: {TRAIN_SCENARIOS}\n")
log_file.write(f"VAL_SCENARIOS: {VAL_SCENARIOS}\n")
log_file.write(f"TEST_SCENARIOS: {TEST_SCENARIOS}\n")
log_file.flush()

 
def main():
    train_dfs = []
    val_dfs = []
    test_dfs = []
    valid_dfs = []
    missing = []

    print("Loading scenarios: ")
    for sid in tqdm.tqdm(TRAIN_SCENARIOS):
        try:
            df = load_scenario_data(sid)
            if df is None or len(df) == 0:
                missing.append(sid)
                continue
            if "Timestamps" in df.columns:
                df = df.drop(columns=["Timestamps"])
            if "Leaks" not in df.columns:
                raise ValueError(f"scenario {sid} missing 'Leaks' column")
            df = add_temporal_columns(df)
            train_dfs.append(df)
            valid_dfs.append(sid)
        except Exception as e:
            print(f"Skiping scenario {sid}: {repr(e)}")
            missing.append(sid)
            continue

    print(f"Loaded Train Scenarios with length {len(train_dfs)} scenarios.")

    for sid in tqdm.tqdm(VAL_SCENARIOS):
        try:
            df = load_scenario_data(sid)
            if df is None or len(df) == 0:
                missing.append(sid)
                continue
            if "Timestamps" in df.columns:
                df = df.drop(columns=["Timestamps"])
            if "Leaks" not in df.columns:
                raise ValueError(f"scenario {sid} missing 'Leaks' column")
            df = add_temporal_columns(df)
            val_dfs.append(df)
            valid_dfs.append(sid)
        except Exception as e:
            print(f"Skiping scenario {sid}: {repr(e)}")
            missing.append(sid)
            continue

    print(f"Loaded Val Scenarios with length {len(val_dfs)} scenarios.")

    for sid in tqdm.tqdm(TEST_SCENARIOS):
        try:
            df = load_scenario_data(sid)
            if df is None or len(df) == 0:
                missing.append(sid)
                continue
            if "Timestamps" in df.columns:
                df = df.drop(columns=["Timestamps"])
            if "Leaks" not in df.columns:
                raise ValueError(f"scenario {sid} missing 'Leaks' column")
            df = add_temporal_columns(df)
            test_dfs.append(df)
            valid_dfs.append(sid)
        except Exception as e:
            print(f"Skiping scenario {sid}: {repr(e)}")
            missing.append(sid)
            continue

    print(f"Loaded Test Scenarios with length {len(test_dfs)} scenarios.")
    log_file.write(f"Missing: {missing}\n")

    print(f"Split: train={len(train_dfs)}, val={len(val_dfs)}, test={len(test_dfs)}")
    log_file.write(f"Split: train={len(train_dfs)}, val={len(val_dfs)}, test={len(test_dfs)}\n")

    mean, std = compute_global_mean_std(train_dfs)
    np.savetxt(os.path.join(files_dir, "mean_lstm.txt"), mean)
    np.savetxt(os.path.join(files_dir, "std_lstm.txt"), std)

    train_dfs = [normalize_df(df, mean, std) for df in train_dfs]
    val_dfs = [normalize_df(df, mean, std) for df in val_dfs]
    test_dfs = [normalize_df(df, mean, std) for df in test_dfs]

    train_ds = WindowedScenarioDatasetSeq(train_dfs, window_size=WINDOW_SIZE, step=1)
    val_ds = WindowedScenarioDatasetSeq(val_dfs, window_size=WINDOW_SIZE, step=1)
    test_ds = WindowedScenarioDatasetSeq(test_dfs, window_size=WINDOW_SIZE, step=1)
    print("Dataset sizes (windows):", len(train_ds), len(val_ds), len(test_ds))
    log_file.write(f"Dataset sizes (windows):, {len(train_ds)}, {len(val_ds)}, {len(test_ds)}\n")

    n_features = train_ds.X.shape[2]  # per-timestep feature count (100), NOT flattened 1200
    print(f"Per-timestep feature count: {n_features}")

    unique, counts = np.unique(train_ds.y, return_counts=True)
    counts_map = dict(zip(unique, counts))
    n_pos = counts_map.get(1, 0)
    n_neg = counts_map.get(0, 0)
    print("Train class counts:", counts_map)
    log_file.write(f"Train class counts: {counts_map}\n")

    weight_for_0 = 1.0 if n_neg == 0 else (n_pos + n_neg) / (2.0 * n_neg)
    weight_for_1 = 1.0 if n_pos == 0 else (n_pos + n_neg) / (2.0 * n_pos)

    pos_weight = torch.tensor(weight_for_1 / weight_for_0).to(DEVICE)

    print(f"Class weights (approx): neg={weight_for_0:.3f}, pos={weight_for_1:.3f}")
    log_file.write(f"Class weights (approx): neg={weight_for_0:.3f}, pos={weight_for_1:.3f}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LeakLSTM(n_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    log_file.write(f"Trainable parameters: {n_params}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    def bce_weighted(preds, targets):
        eps = 1e-7
        preds = torch.clamp(preds, eps, 1 - eps)
        weights = torch.where(targets == 1, torch.tensor(weight_for_1, device=DEVICE), torch.tensor(weight_for_0, device=DEVICE))
        loss = - (weights * (targets.float() * torch.log(preds) + (1 - targets.float()) * torch.log(1 - preds)))
        return loss.mean()

    best_val_loss = 1000000
    best_val_f1 = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm.tqdm(train_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            loss = bce_weighted(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * (xb.size(0))

        train_loss = running_loss / len(train_ds)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        train_preds_all = []
        train_labs_all = []
        val_preds_all = []
        val_labs_all = []

        with torch.no_grad():
            for xb, yb in tqdm.tqdm(val_loader):
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = bce_weighted(preds, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds_all.append(preds.cpu().numpy())
                val_labs_all.append(yb.cpu().numpy())

            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = bce_weighted(preds, yb)
                train_preds_all.append(preds.cpu().numpy())
                train_labs_all.append(yb.cpu().numpy())

        val_loss = val_loss / len(val_ds)
        val_preds_all = np.concatenate(val_preds_all)
        val_labs_all = np.concatenate(val_labs_all)
        val_pred_labels = (val_preds_all >= 0.5).astype(int)
        p, r, val_f1, _ = precision_recall_fscore_support(val_labs_all, val_pred_labels, average='binary', zero_division=0)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        train_preds_all = np.concatenate(train_preds_all)
        train_labs_all = np.concatenate(train_labs_all)
        train_pred_labels = (train_preds_all >= 0.5).astype(int)
        p, r, train_f1, _ = precision_recall_fscore_support(train_labs_all, train_pred_labels, average='binary', zero_division=0)
        history["train_f1"].append(train_f1)

        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f} | train_f1: {train_f1:.6f} | val_loss: {val_loss:.6f} | val_f1: {val_f1:.4f}")
        log_file.write(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f} | train_f1: {train_f1:.6f} | val_loss: {val_loss:.6f} | val_f1: {val_f1:.4f}\n")

        scheduler.step(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(files_dir, f"_lstm_last_{id}_{n_scenarios}_{WINDOW_SIZE}_ablation.pth"))
            print(f"Saved Model at Epoch {epoch}")

    print("Training Finished. Best Val F1: ", best_val_f1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history["val_f1"], label="val_f1")
    plt.plot(history["train_f1"], label="train_f1")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.legend()
    plt.title("F1")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"training_curves_lstm_{id}.png"))
    plt.show()

    model.load_state_dict(torch.load(os.path.join(files_dir, f"_lstm_last_{id}_{n_scenarios}_{WINDOW_SIZE}_ablation.pth")))
    model.eval()

    preds_all = []
    labs_all = []
    with torch.no_grad():
        for xb, yb in tqdm.tqdm(test_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            preds_all.append(preds.cpu().numpy())
            labs_all.append(yb.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labs_all = np.concatenate(labs_all)

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(labs_all, preds_all)
    roc_auc = roc_auc_score(labs_all, preds_all)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    log_file.write(f"Best Threshold: {best_threshold}\n")

    print("Best threshold =", best_threshold)
    print("ROC-AUC =", roc_auc)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"roc_curve_test_lstm_{id}.png"))
    plt.show()

    pred_labels = (preds_all >= best_threshold).astype(int)

    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average='binary', zero_division=0)
    cm = confusion_matrix(labs_all, pred_labels)

    print("Test metrics (best_threshold): acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".format(acc, p, r, f1))
    print("Confusion matrix:\n", cm)
    log_file.write("Test metrics (best_threshold={:.4f}): acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}\n".format(best_threshold, acc, p, r, f1))

 
main()