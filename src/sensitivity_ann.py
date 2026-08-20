 
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
    WORKDIR = os.path.dirname(CURRENT_PATH)
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
    

 
TEST_SCENARIOS_1 = [3, 5, 11, 30, 49, 61, 64, 71, 77, 78, 80, 89, 93, 108, 109, 122, 123]
TEST_SCENARIOS_2 = [127, 140, 143, 156, 160, 164, 183, 190, 192, 231, 239, 248, 257, 262, 272 ]  
TEST_SCENARIOS_3 = [286, 295, 300, 306, 311, 320, 325, 331, 332, 333, 351, 370, 430, 436, 447]  
TEST_SCENARIOS_4 = [448, 466, 501, 533, 534, 539, 543, 557, 563, 573, 574, 580, 610, 612, 618 ]
TEST_SCENARIOS_5 = [626, 642, 644, 651, 668, 673, 677, 683, 686, 690, 691, 692, 694, 699, 722 ] 
TEST_SCENARIOS_6 = [725, 731, 732, 754, 755, 780, 787, 812, 814, 815, 835, 854, 862, 866, 867, 873, 875, 887, 910, 914, 930, 951, 952, 963, 965, 981, 994, 995 ]

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FILES_PATH = os.path.join(CURRENT_PATH, "files")
MODEL_PATH = os.path.join(FILES_PATH, r"ann_16_2026_01_13_15_20_21.pth")
MEAN_PATH = os.path.join(FILES_PATH, r"mean_ann.txt")
STD_PATH = os.path.join(FILES_PATH, r"std_ann.txt")

 
import gc

def test(alpha, beta):
    model = LeakANN(1200).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    mean = np.loadtxt(MEAN_PATH)
    std = np.loadtxt(STD_PATH)

    preds_all = []
    labs_all = []

    scenario_map = {
        1: TEST_SCENARIOS_1, 2: TEST_SCENARIOS_2, 3: TEST_SCENARIOS_3,
        4: TEST_SCENARIOS_4, 5: TEST_SCENARIOS_5, 6: TEST_SCENARIOS_6
    }

    for i in range(1, 7):
        current_scenarios = scenario_map.get(i, [])
        print(f"Processing Batch {i}...")

        for sid in tqdm.tqdm(current_scenarios):
            try:
                df = load_scenario_data(sid)
                if df is None or len(df) == 0: continue
                
                df = scale_data(df, alpha, beta)
                
                if "Timestamps" in df.columns:
                    df = df.drop(columns=["Timestamps"])
                
                df = add_temporal_columns(df)
                df = normalize_df(df, mean, std) 

                test_ds = WindowedScenarioDataset([df], window_size=WINDOW_SIZE, step=1)
                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

                with torch.no_grad():
                    for xb, yb in test_loader:
                        preds = model(xb.to(DEVICE))
                        preds_all.append(preds.cpu().numpy())
                        labs_all.append(yb.cpu().numpy())

                del df
                del test_ds
                del test_loader

            except Exception as e:
                print(f"Skipping scenario {sid}: {repr(e)}")
                continue
        
        gc.collect()

    preds_all = np.concatenate(preds_all)
    labs_all = np.concatenate(labs_all)

    pred_labels = (preds_all >= 0.5).astype(int)
    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average='binary', zero_division=0)
    
    print(f"Test metrics: acc = {acc:.4f}, precision = {p:.4f}, recall = {r:.4f}, f1 = {f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(labs_all, pred_labels))

 
import gc
import torch
import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_single_test_config(alpha, beta, model, mean, std):
    """Performs inference for all scenarios using specific alpha/beta scaling."""
    preds_all = []
    labs_all = []
    
    all_test_groups = [
        TEST_SCENARIOS_1, TEST_SCENARIOS_2, TEST_SCENARIOS_3,
        TEST_SCENARIOS_4, TEST_SCENARIOS_5, TEST_SCENARIOS_6
    ]

    for group in all_test_groups:
        for sid in group:
            try:
                df = load_scenario_data(sid)
                if df is None or len(df) == 0: continue
                if "Timestamps" in df.columns:
                    df = df.drop(columns=["Timestamps"])
                
                df = scale_data(df, alpha, beta)
                df = df.copy() 
                
                df = add_temporal_columns(df)
                df = normalize_df(df, mean, std)

                test_ds = WindowedScenarioDataset([df], window_size=WINDOW_SIZE, step=1)
                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

                with torch.no_grad():
                    for xb, yb in test_loader:
                        preds = model(xb.to(DEVICE))
                        preds_all.append(preds.cpu().numpy())
                        labs_all.append(yb.cpu().numpy())
                
                del df, test_ds, test_loader
            except Exception as e:
                print(f"Error in scenario {sid}: {e}")
                continue
        gc.collect() 

    if not preds_all: return None
    
    preds_all = np.concatenate(preds_all)
    labs_all = np.concatenate(labs_all)
    pred_labels = (preds_all >= 0.5).astype(int)
    
    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average='binary', zero_division=0)
    
    return {"alpha": alpha, "beta": beta, "accuracy": acc, "f1": f1, "precision": p, "recall": r}

metrics = run_single_test_config(a, b, model, mean, std)
print(metrics)

 
import seaborn as sns
import matplotlib.pyplot as plt

alphas = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
betas = [0.2, 0.4, 0.6, 0.8, 1.0]
results = []

model = LeakANN(1200).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
mean = np.loadtxt(MEAN_PATH)
std = np.loadtxt(STD_PATH)

for a in alphas:
    for b in betas:
        print(f"Testing Alpha: {a}, Beta: {b}...")
        metrics = run_single_test_config(a, b, model, mean, std)
        if metrics:
            results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv("leak_detection_sensitivity.csv", index=False)

 
def plot_sensitivity(df, metric="f1"):
    pivot_table = df.pivot(index="beta", columns="alpha", values=metric)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f"Model Robustness: {metric.upper()} Score")
    plt.xlabel("Alpha (Demand/Flow Shift)")
    plt.ylabel("Beta (Pressure Sensitivity)")
    plt.show()

plot_sensitivity(results_df, metric="f1")

 



