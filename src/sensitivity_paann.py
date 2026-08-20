 
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

    data  = pd.read_csv(os.path.join(WORKDIR, r"leakdb_Hanoi\Full-Dataset", f"Scenario-{scenario_id}.csv"))
    data = data.drop(columns=['Unnamed: 0'])
    return data

 
def add_temporal_columns(df):
    
    step_of_day = np.array([i % 48 for i in range(len(df))])
    
    df = df.assign(sin_hour = np.sin(2 * np.pi * step_of_day / 48))
    df = df.assign(cos_hour =np.cos(2 * np.pi * step_of_day / 48))
    
    return df

 
def scale_data(df, alpha, beta=0.7):
    demand_cols = [col for col in df.columns if col.startswith('demand_node_')]
    df[demand_cols] = df[demand_cols] * (1 + alpha)
    flow_cols = [col for col in df.columns if col.startswith('flow_link_')]
    df[flow_cols] = df[flow_cols] * (1 + alpha)
    pressure_cols = [col for col in df.columns if col.startswith('pressure_node_')]
    df[pressure_cols] = df[pressure_cols] * (1 - (beta * alpha))
    
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
    def __init__(self, df_list, scenario_ids, window_size=5, step=1):
        self.window_size = window_size
        self.step = step
        self.X = []
        self.y = []
        self.scenario_ids = []
        self.demands = []
        
        for df, sid in zip(df_list, scenario_ids):
            arr = df.drop(columns=["Leaks"]).values.astype(np.float32)  
            labs = df["Leaks"].values.astype(np.int64)                
            T, F = arr.shape
            for start in range(0, T - window_size + 1, step):
                window = arr[start:start+window_size].reshape(-1)     
                target = labs[start + window_size - 1]            
                self.X.append(window)
                self.y.append(target)
                self.scenario_ids.append(sid)
                self.demands.append(
                    arr[start + window_size - 1, 0:32]
                )
                
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.scenario_ids[idx], torch.tensor(self.demands[idx])

 
class LeakPhysicsANN(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=256, hidden3=64, n_nodes=32, n_pipes=34):
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
        
        self.pressure_head = nn.Linear(hidden3, n_nodes)
        self.flow_head = nn.Linear(hidden3, n_pipes)
    
        self.leak_head = nn.Sequential(
            nn.Linear(hidden3 + 5, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        
        pred_p = self.pressure_head(h)
        pred_f = self.flow_head(h)
        
        p_mean = pred_p.mean(dim=1)
        p_std = pred_p.std(dim=1)
        p_rng = pred_p.max(dim=1).values - pred_p.min(dim=1).values
        
        f_mean = pred_f.mean(dim=1)
        f_std = pred_f.std(dim=1)
        
        phys_feat = torch.stack(
            [p_mean, p_std, p_rng, f_mean, f_std], dim=1
        )
        
        leak_input = torch.cat([h, phys_feat], dim=1)
        leak_prob = torch.sigmoid(self.leak_head(leak_input)).squeeze(1)
        
        return leak_prob, pred_p, pred_f

 
def normalize_df(df, mean, std):
    cols = [c for c in df.columns if c != "Leaks"]
    df2 = df.copy()
    df2[cols] = (df2[cols] - mean) / std
    return df2

 
TEST_SCENARIOS_1 = [3, 5, 11, 30, 49, 61, 64, 71, 77, 78, 80, 89, 93, 108, 109, 122, 123]
TEST_SCENARIOS_2 = [127, 140, 143, 156, 160, 164, 183, 190, 192, 231, 239, 248, 257, 262, 272 ]  
TEST_SCENARIOS_3 = [286, 295, 300, 306, 311, 320, 325, 331, 332, 333, 351, 370, 430, 436, 447]  
TEST_SCENARIOS_4 = [448, 466, 501, 533, 534, 539, 543, 557, 563, 573, 574, 580, 610, 612, 618 ]
TEST_SCENARIOS_5 = [626, 642, 644, 651, 668, 673, 677, 683, 686, 690, 691, 692, 694, 699, 722 ] 
TEST_SCENARIOS_6 = [725, 731, 732, 754, 755, 780, 787, 812, 814, 815, 835, 854, 862, 866, 867, 873, 875, 887, 910, 914, 930, 951, 952, 963, 965, 981, 994, 995 ]
TEST_SCENARIOS = TEST_SCENARIOS_1 + TEST_SCENARIOS_2+ TEST_SCENARIOS_3+ TEST_SCENARIOS_4+ TEST_SCENARIOS_5+ TEST_SCENARIOS_6


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FILES_PATH = os.path.join(CURRENT_PATH, "files")

MODEL_PATH = os.path.join(FILES_PATH, r"pann_best.pth")
MEAN_PATH = os.path.join(FILES_PATH, r"mean_ann_physics.txt")
STD_PATH = os.path.join(FILES_PATH, r"std_ann_physics.txt")

 
def test(alpha, beta, mean, std, model):
    
    test_dfs = []
    
    for sid in tqdm.tqdm(TEST_SCENARIOS):
        
        try:
            df = load_scenario_data(sid)
            df = scale_data(df, alpha, beta)
            if "Timestamps" in df.columns:
                df = df.drop(columns=["Timestamps"])
            df = add_temporal_columns(df)
            test_dfs.append(df)
        except Exception as e:
            print(f"Skiping scenario {sid}: {repr(e)}")
            continue
    
    test_dfs = [normalize_df(df, mean, std) for df in test_dfs]
    test_ds = WindowedScenarioDataset(test_dfs, scenario_ids=TEST_SCENARIOS, window_size=WINDOW_SIZE, step=1)
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = LeakPhysicsANN(1200).to(DEVICE)
    
    from sklearn.metrics import precision_recall_curve, average_precision_score, auc
    
    preds_all = []
    labs_all = []
    with torch.no_grad():
        for xb, yb, sid_batch, true_d in tqdm.tqdm(test_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds, _, _ = model(xb)
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
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm
    }

   
    

 
import gc
import torch
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader

def run_single_test_config(
    alpha,
    beta,
    model,
    mean,
    std,
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE
):
    preds_all = []
    labs_all = []

    all_test_groups = [
        TEST_SCENARIOS_1, TEST_SCENARIOS_2, TEST_SCENARIOS_3,
        TEST_SCENARIOS_4, TEST_SCENARIOS_5, TEST_SCENARIOS_6
    ]

    model.eval()

    for group in all_test_groups:
        test_dfs = []
        scenario_ids = []

        for sid in group:
            try:
                df = load_scenario_data(sid)
                df = scale_data(df, alpha, beta) 
                if df is None or len(df) == 0:
                    continue

                if "Timestamps" in df.columns:
                    df = df.drop(columns=["Timestamps"])

                if "Leaks" not in df.columns:
                    raise ValueError(f"Scenario {sid} missing 'Leaks' column")

                df = add_temporal_columns(df)
                df = normalize_df(df, mean, std)

                test_dfs.append(df)
                scenario_ids.append(sid)

            except Exception as e:
                print(f"Skipping scenario {sid}: {e}")
                continue

        if not test_dfs:
            continue

        test_ds = WindowedScenarioDataset(
            test_dfs,
            scenario_ids=scenario_ids,
            window_size=window_size,
            step=1
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for xb, yb, sid_batch, true_d in tqdm.tqdm(test_loader, leave=False):
                xb = xb.to(device)
                yb = yb.to(device)

                preds, _, _ = model(xb)

                preds_all.append(preds.cpu().numpy())
                labs_all.append(yb.cpu().numpy())

        del test_dfs, test_ds, test_loader
        gc.collect()

    if not preds_all:
        return None

    preds_all = np.concatenate(preds_all)
    labs_all = np.concatenate(labs_all)

    pred_labels = (preds_all >= 0.5).astype(int)

    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(
        labs_all,
        pred_labels,
        average="binary",
        zero_division=0
    )
    cm = confusion_matrix(labs_all, pred_labels)

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm
    }

 
import gc
import torch
import numpy as np
import pandas as pd
import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class WindowedScenarioDataset(Dataset):
    def __init__(self, df_list, scenario_ids, window_size=5, step=1):
        self.window_size = window_size
        self.step = step
        self.X = []
        self.y = []
        self.scenario_ids = []
        self.demands = []
        
        for df, sid in zip(df_list, scenario_ids):
            cols_to_drop = ["Leaks", "Timestamps"]
            drop_cols = [c for c in cols_to_drop if c in df.columns]
            
            arr = df.drop(columns=drop_cols).values.astype(np.float32)  
            labs = df["Leaks"].values.astype(np.int64)                
            
            T, F = arr.shape
            for start in range(0, T - window_size + 1, step):
                window = arr[start:start+window_size].reshape(-1)     
                target = labs[start + window_size - 1]            
                
                self.X.append(window)
                self.y.append(target)
                self.scenario_ids.append(sid)
                self.demands.append(arr[start + window_size - 1, 0:32])
                
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.scenario_ids[idx], torch.tensor(self.demands[idx])

def test(alpha, beta, mean, std, model):
    preds_all = []
    labs_all = []
    
    for sid in tqdm.tqdm(TEST_SCENARIOS, desc=f"α={alpha}, β={beta}", leave=False):
        try:
            df = load_scenario_data(sid)
            if df is None or len(df) == 0: continue
            if "Timestamps" in df.columns:
                df = df.drop(columns=["Timestamps"])
            
            df = scale_data(df, alpha, beta)
            df = df.copy() 
            df = add_temporal_columns(df)
            df = normalize_df(df, mean, std)
            
            mini_ds = WindowedScenarioDataset([df], [sid], window_size=WINDOW_SIZE, step=1)
            mini_loader = DataLoader(mini_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            with torch.no_grad():
                for xb, yb, _, _ in mini_loader:
                    xb = xb.to(DEVICE)
                    
                    probs, _, _ = model(xb)
                    
                    preds_all.append(probs.cpu().numpy())
                    labs_all.append(yb.numpy())

            del df, mini_ds, mini_loader
            
        except Exception as e:
            print(f"Skipping scenario {sid}: {repr(e)}")
            continue

    gc.collect() 
    
    if not preds_all: return None

    preds_all = np.concatenate(preds_all)
    labs_all = np.concatenate(labs_all)
    
    pred_labels = (preds_all >= 0.5).astype(int)

    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average='binary', zero_division=0)
    cm = confusion_matrix(labs_all, pred_labels)
    
    return {
        "alpha": alpha, "beta": beta,
        "accuracy": acc, "precision": p, "recall": r, "f1": f1
    }

import seaborn as sns
import matplotlib.pyplot as plt

alphas = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
betas = [0.2, 0.4, 0.6, 0.8, 1.0]
results = []

model = LeakPhysicsANN(1200).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

mean = np.loadtxt(MEAN_PATH)
std = np.loadtxt(STD_PATH)

print("Starting Sensitivity Analysis...")
print(f"{'Alpha':>8} | {'Beta':>8} | {'Accuracy':>10} | {'F1':>10}")
print("-" * 45)

for a in alphas:
    for b in betas:
        metrics = test(a, b, mean, std, model) 
        
        if metrics:
            results.append(metrics)
            print(f"{a:8.2f} | {b:8.2f} | {metrics['accuracy']:10.4f} | {metrics['f1']:10.4f}")
        
        gc.collect()

results_df = pd.DataFrame(results)
results_df.to_csv("leak_detection_sensitivity_physics.csv", index=False)

 
def plot_sensitivity(df, metric="f1"):
    pivot_table = df.pivot(index="beta", columns="alpha", values=metric)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f"Model Robustness: {metric.upper()} Score")
    plt.xlabel("Alpha (Demand/Flow Shift)")
    plt.ylabel("Beta (Pressure Sensitivity)")
    plt.show()

plot_sensitivity(results_df, metric="f1")


