
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
import os

 
def load_scenario_data(scenario_id):

    data  = pd.read_csv(os.path.join("scenario_folder_path", f"Scenario-{scenario_id}.csv"))
    data = data.drop(columns=['Unnamed: 0'])

    return data

 
def add_temporal_columns(df):
    
    step_of_day = np.array([i % 48 for i in range(len(df))])
    
    df["sin_hour"] = np.sin(2 * np.pi * step_of_day / 48)
    df["cos_hour"] = np.cos(2 * np.pi * step_of_day / 48)
    
    return df

 
WINDOW_SIZE = 12
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 30
DROPOUT = 0.5
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
class WindowedScenarioDataset(Dataset):
    def __init__(self, df_list, window_size=5, step=1):
        self.window_size = window_size
        self.step = step
        self.X = []
        self.y = []
        for df in df_list:
            arr = df.drop(columns=["Leaks"]).values.astype(np.float32)  
            labs = df["Leaks"].values.astype(np.int64)                
            T, F = arr.shape
            for start in range(0, T - window_size + 1, step):
                window = arr[start:start+window_size].reshape(-1)     
                target = labs[start + window_size - 1]            
                self.X.append(window)
                self.y.append(target)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

 
class LeakANN(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=256, hidden3=64):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU()
        )
        
        self.out = nn.Linear(hidden3, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return torch.sigmoid(self.out(x)).squeeze(1)  

 
def normalize_df(df, mean, std):
    cols = [c for c in df.columns if c != "Leaks"]
    df2 = df.copy()
    df2[cols] = (df2[cols] - mean) / std
    return df2
    
TEST_SCENARIOS = []

 
def test():
    
    test_dfs = []
    
    for sid in tqdm.tqdm(TEST_SCENARIOS):
        
        try:
            df = load_scenario_data(sid)
            if df is None or len(df) == 0:
                continue
            
            if "Timestamps" in df.columns:
                df = df.drop(columns=["Timestamps"])
                
            if "Leaks" not in df.columns:
                raise ValueError(f"scenario {sid} missing 'Leaks' column")

            df = add_temporal_columns(df)

            test_dfs.append(df)

        except Exception as e:
            print(f"Skiping scenario {sid}: {repr(e)}")
            continue
        
    mean = np.loadtxt("path_of_mean_ann.txt")
    std = np.loadtxt("path_of_std_ann.txt")
    
    test_dfs = [normalize_df(df, mean, std) for df in test_dfs]
    test_ds = WindowedScenarioDataset(test_dfs, window_size=WINDOW_SIZE, step=1)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model  = LeakANN(1200).to(DEVICE)
    
    model.load_state_dict(torch.load(os.path.join(r"path_of_best_ann.pth")))
    model.eval()
    
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    
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
    
    pred_labels = (preds_all >= 0.5).astype(int)

    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average='binary', zero_division=0)
    cm = confusion_matrix(labs_all, pred_labels)

    print("Test metrics: acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".format(acc, p, r, f1))
    print("Confusion matrix:\n", cm)
    print("="*20)

 
def test_roc_auc():
    
    test_dfs = []
    
    for sid in tqdm.tqdm(TEST_SCENARIOS):
        
        try:
            df = load_scenario_data(sid)
            if df is None or len(df) == 0:
                continue
            
            if "Timestamps" in df.columns:
                df = df.drop(columns=["Timestamps"])
                
            if "Leaks" not in df.columns:
                raise ValueError(f"scenario {sid} missing 'Leaks' column")

            df = add_temporal_columns(df)

            test_dfs.append(df)

        except Exception as e:
            print(f"Skiping scenario {sid}: {repr(e)}")
            continue
        
    mean = np.loadtxt("path_of_mean_ann.txt")
    std = np.loadtxt("path_of_std_ann.txt")
    
    test_dfs = [normalize_df(df, mean, std) for df in test_dfs]
    test_ds = WindowedScenarioDataset(test_dfs, window_size=WINDOW_SIZE, step=1)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model  = LeakANN(1200).to(DEVICE)
    
    model.load_state_dict(torch.load(os.path.join(r"path_of_best_ann.pth")))
    model.eval()
    
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    
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
    plt.show()
    
    pred_labels = (preds_all >= best_threshold).astype(int)

    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average='binary', zero_division=0)
    cm = confusion_matrix(labs_all, pred_labels)

    print("Test metrics: acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".format(acc, p, r, f1))
    print("Confusion matrix:\n", cm)

print("On Threshold = 0.5\n")
test()


