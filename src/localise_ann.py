import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import random
import tqdm

WINDOW_SIZE = 12 
THRESHOLD = 0.5  
BASELINE_WINDOW = 20 
DROPOUT = 0.5
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_dir = os.getcwd()
files_dir = os.path.join(current_dir, "files")
dataset_dir = os.path.join(current_dir, "..","..", "Hanoi_CMH")
MODEL_PATH = os.path.join(files_dir, "_ann_last_2026_01_13_15_20_21.pth") 


print(f"Dataset Dir: {dataset_dir}")
print(f"Model Path: {MODEL_PATH}")
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

import os
import pandas as pd
import numpy as np

def load_normalization_stats(files_dir):
    mean = np.loadtxt(os.path.join(files_dir, "mean_ann.txt")).astype(np.float32)
    std = np.loadtxt(os.path.join(files_dir, "std_ann.txt")).astype(np.float32)
    return mean, std

def load_full_scenario(scenario_id, dataset_dir):
    s_path = os.path.join(dataset_dir, f"Scenario-{scenario_id}")
    if not os.path.exists(s_path): return None, None

    def get_combined(subfolder, prefix):
        p = os.path.join(s_path, subfolder)
        if not os.path.exists(p): return pd.DataFrame()
        files = sorted([f for f in os.listdir(p) if f.endswith(".csv")])
        dfs = []
        for f in files:
            df = pd.read_csv(os.path.join(p, f)).drop(columns="Index", errors="ignore")
            df.columns = [f"{prefix}_{f.replace('.csv', '')}"]
            dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    full_df = pd.concat([
        get_combined("Demands", "demand"),
        get_combined("Pressures", "pressure"),
        get_combined("Flows", "flow")
    ], axis=1)

    step_of_day = np.arange(len(full_df)) % 48
    full_df["sin_hour"] = np.sin(2 * np.pi * step_of_day / 48)
    full_df["cos_hour"] = np.cos(2 * np.pi * step_of_day / 48)
    
    pressure_only = get_combined("Pressures", "pressure")
    
    return full_df, pressure_only

def load_ground_truth(scenario_id, dataset_dir):

    leaks_path = os.path.join(dataset_dir, f"Scenario-{scenario_id}", "Leaks")
   
    if not os.path.exists(leaks_path):
        print(f"Warning: Leaks folder not found at {leaks_path}")
        return []
    
    info_files = [f for f in os.listdir(leaks_path) if f.endswith("_info.csv")]
    
    if not info_files:
        print("No leak info files found.")
        return []
        
    ground_truth_list = []
    
    for f in info_files:
        file_path = os.path.join(leaks_path, f)
        try:
            df = pd.read_csv(file_path)
            
            df.columns = [c.strip() for c in df.columns]
            
            data_map = dict(zip(df['Description'].astype(str).str.strip(), df['Value']))
            
            leak_entry = {
                "node": str(data_map.get("Leak Node", "?")),  # e.g., "6"
                "start": int(data_map.get("Leak Start", 0)),  # e.g., 4881
                "end": int(data_map.get("Leak End", 0)),      # e.g., 8994
                "type": str(data_map.get("Leak Type", "unknown")),
                "filename": f
            }
            
            ground_truth_list.append(leak_entry)
            
        except Exception as e:
            print(f"Error parsing file {f}: {e}")
            
    print(f"Loaded {len(ground_truth_list)} leak(s) from ground truth.")
    return ground_truth_list

def normalize_data(df, mean, std):
    
    data = df.values.astype(np.float32)
    
    epsilon = 1e-6

    norm_data = (data - mean) / (std + epsilon)

    return pd.DataFrame(norm_data, columns=df.columns)

import time
import numpy as np
import torch
from collections import deque

NETWORK_GRAPH = {
    '1': ['2'], '2': ['1', '3'], '3': ['2', '4', '19'],
    '4': ['3', '5'], '5': ['4', '6'], '6': ['5', '7'],
    '7': ['6', '8'], '8': ['7', '9'], '9': ['8', '10', '14'],
    '10': ['9', '11', '14'], '11': ['10', '12'], '12': ['11', '13'],
    '13': ['12'], '14': ['9', '15', '10'], '15': ['14', '16'], 
    '16': ['15', '17', '27'], '17': ['16', '18'], '18': ['17', '19'], 
    '19': ['18', '3'], '20': ['23', '21'], '21': ['20', '22'], 
    '22': ['21'], '23': ['28', '20', '24'], '24': ['23', '25'], 
    '25': ['24', '26', '32'], '26': ['25', '27'], '27': ['26', '16'], 
    '28': ['29', '23'], '29': ['28', '30'], '30': ['29', '31'], 
    '31': ['30', '32'], '32': ['31', '25']
}

def get_topology_score(node_id, all_z_scores, active_baseline_names):
    
    try:
        idx = -1
        for i, name in enumerate(active_baseline_names):
            if name.endswith(f"_{node_id}"):
                idx = i; break
        if idx == -1: return 0.0
        
        own_z = all_z_scores[idx]
        neighbors = NETWORK_GRAPH.get(str(node_id), [])

        neighbor_z = []
        for n_id in neighbors:
            for i, name in enumerate(active_baseline_names):
                if name.endswith(f"_{n_id}"):
                    neighbor_z.append(all_z_scores[i]); break
        
        support_ratio = 0.0
        if neighbor_z and own_z > 0:
            support_ratio = min(1.0, np.mean(neighbor_z) / own_z)

        degree_bonus = 1.0 + (0.05 * len(neighbors))
        
        return own_z * (1.0 + support_ratio) * degree_bonus
    except:
        return 0.0

def run_realtime_dashboard(full_data, pressure_data, model, device):
    print(f"--- INITIALIZING PURE ANN MONITORING ---")
    
    ann_window = deque(maxlen=WINDOW_SIZE)
    stats_buffer = deque(maxlen=5000) 
    detected_history = []
    prob_history = []
    
    CONFIDENCE_TRIGGER = 0.75  
    CONFIDENCE_RESET = 0.50    
    BASELINE_LAG = 50          
    BASELINE_WINDOW = 200      

    leak_active = False
    leak_start_time = 0
    leak_type = "Scanning"
    
    smoothed_prob = 0.0
    active_baseline = None
    vote_buffer = {}
    
    current_winner = None
    winner_stability_counter = 0
    is_locked = False
    
    for t in range(len(full_data)):
        row = full_data.iloc[t].values.astype(np.float32)
        p_row = pressure_data.iloc[t]
        ann_window.append(row)

        raw_prob = 0.0
        if len(ann_window) == WINDOW_SIZE:
            inp = torch.tensor(np.array(ann_window).flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                raw_prob = model(inp).item()
            
            if not leak_active and raw_prob < 0.6: 
                stats_buffer.append(p_row.values)

        if raw_prob > smoothed_prob: smoothed_prob = 0.7 * raw_prob + 0.3 * smoothed_prob
        else: smoothed_prob = 0.1 * raw_prob + 0.9 * smoothed_prob
        prob_history.append(smoothed_prob)

        if not leak_active:
            if smoothed_prob > CONFIDENCE_TRIGGER:
                leak_active = True
                leak_start_time = t
                leak_type = "Analyz"
                
                vote_buffer = {}
                current_winner = None
                winner_stability_counter = 0
                is_locked = False
                
                if len(stats_buffer) > (BASELINE_LAG + 10):
                    buff_snap = np.array(stats_buffer)
                    clean_slice = buff_snap[max(0, len(stats_buffer)-(BASELINE_LAG+BASELINE_WINDOW)) : -BASELINE_LAG]
                    active_baseline = {
                        "mean": np.median(clean_slice, axis=0),
                        "std": np.maximum(np.std(clean_slice, axis=0), 0.05),
                        "names": pressure_data.columns
                    }
        else:
            if smoothed_prob < CONFIDENCE_RESET:
                leak_active = False
                active_baseline = None
                leak_type = "Scanning"

        display_suspects = "Scanning..."
        max_drop = 0.0
        
        if leak_active and active_baseline:
            curr_vals = p_row.values
            raw_drops = active_baseline["mean"] - curr_vals
            valid_drops = np.maximum(0, raw_drops)
            max_drop = np.max(valid_drops)
            z_scores = valid_drops / active_baseline["std"]

            duration = t - leak_start_time
            time_weight = 1.0 / (1.0 + 0.15 * duration) 

            if not is_locked:
                topo_candidates = []
                for i, name in enumerate(active_baseline["names"]):
                    if z_scores[i] > 1.5: 
                        node_id = name.split('_')[-1]
                        score = get_topology_score(node_id, z_scores, active_baseline["names"])
                        topo_candidates.append((node_id, score))
                
                topo_candidates.sort(key=lambda x: x[1], reverse=True)

                for rank, (node, score) in enumerate(topo_candidates[:3]):
                    points = (3 - rank) * time_weight
                    vote_buffer[node] = vote_buffer.get(node, 0) + points
                
                sorted_votes = sorted(vote_buffer.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_votes:
                    new_winner = sorted_votes[0][0]
                    
                    if new_winner == current_winner:
                        winner_stability_counter += 1
                        if winner_stability_counter > 8: 
                            is_locked = True
                            leak_type = "LOCKED"
                    else:
                        current_winner = new_winner
                        winner_stability_counter = 0 
                    
                    display_suspects = str([n for n, p in sorted_votes[:3]])
            
            else:
                display_suspects = f"['{current_winner}'] (FIXED)"
                leak_type = "LOCKED"

        icon = "🟢"
        if leak_active: icon = "🔴"
        elif smoothed_prob > 0.4: icon = "🟠"
        
        info = ""
        if leak_active:
            info = f"| Suspects: {display_suspects:<30} | MaxDrop: {max_drop:.1f}m"
        
        if t % 1 == 0 and (leak_active or t % 100 == 0): 
             print(f"T: {t:<5} | {icon} {leak_type:<5} | Conf: {smoothed_prob*100:4.1f}% {info}")
        
        if leak_active:
            suspect = current_winner if current_winner else "?"
            detected_history.append({'start': t, 'end': t+1, 'suspect': suspect})
            
    return detected_history, prob_history

SCENARIO_ID = 32

mean_stats, std_stats = load_normalization_stats(files_dir)
ann_df, p_df = load_full_scenario(SCENARIO_ID, dataset_dir)
ann_norm = (ann_df - mean_stats) / std_stats

model = LeakANN(ann_norm.shape[1] * WINDOW_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Running Simulation...")
events, probs = run_realtime_dashboard(ann_norm, p_df, model, device)
print("Done.")

def run_silent_evaluation(full_data, pressure_data, model, device, gt_leaks=None):
    ann_window = deque(maxlen=WINDOW_SIZE)
    stats_buffer = deque(maxlen=5000)
    
    detected_events = []
    prob_history = [] 
    
    CONFIDENCE_TRIGGER = 0.75
    CONFIDENCE_RESET = 0.50
    BASELINE_LAG = 50; BASELINE_WINDOW = 200
    
    leak_active = False
    leak_start_time = 0
    smoothed_prob = 0.0
    active_baseline = None
    vote_buffer = {}
    
    current_winner = None
    winner_stability_counter = 0
    is_locked = False
    
    for t in range(len(full_data)):
        row = full_data.iloc[t].values.astype(np.float32)
        p_row = pressure_data.iloc[t]
        ann_window.append(row)
        
        raw_prob = 0.0
        if len(ann_window) == WINDOW_SIZE:
            inp = torch.tensor(np.array(ann_window).flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                raw_prob = model(inp).item()
            
            if not leak_active and raw_prob < 0.6: 
                stats_buffer.append(p_row.values)

        if raw_prob > smoothed_prob: smoothed_prob = 0.7 * raw_prob + 0.3 * smoothed_prob
        else: smoothed_prob = 0.1 * raw_prob + 0.9 * smoothed_prob
        
        prob_history.append(smoothed_prob)

        if not leak_active:
            if smoothed_prob > CONFIDENCE_TRIGGER:
                leak_active = True
                leak_start_time = t
                vote_buffer = {}; current_winner = None; winner_stability_counter = 0; is_locked = False
         
                curr_len = len(stats_buffer)
                if curr_len > (BASELINE_LAG + 10):
                    buff_snap = np.array(stats_buffer)
                    clean = buff_snap[max(0, curr_len-(BASELINE_LAG+BASELINE_WINDOW)):-BASELINE_LAG]
                    active_baseline = {
                        "mean": np.median(clean, axis=0), 
                        "std": np.maximum(np.std(clean, axis=0), 0.05), 
                        "names": pressure_data.columns
                    }
        else:
            if smoothed_prob < CONFIDENCE_RESET:
                leak_active = False; active_baseline = None; is_locked = False

        final_suspect = None
        if leak_active and active_baseline:
            curr_vals = p_row.values
            drops = np.maximum(0, active_baseline["mean"] - curr_vals)
            z_scores = drops / active_baseline["std"]
            
            if is_locked:
                final_suspect = current_winner
            else:
                duration = t - leak_start_time
                time_weight = 1.0 / (1.0 + 0.15 * duration)
                
                cands = []
                for i, name in enumerate(active_baseline["names"]):
                    if z_scores[i] > 1.5: 
                        nid = name.split('_')[-1]
                        cands.append((nid, get_topology_score(nid, z_scores, active_baseline["names"])))
                cands.sort(key=lambda x:x[1], reverse=True)
                
                for r, (n, s) in enumerate(cands[:3]): 
                    vote_buffer[n] = vote_buffer.get(n, 0) + ((3 - r) * time_weight)
                
                sorted_votes = sorted(vote_buffer.items(), key=lambda x:x[1], reverse=True)
                if sorted_votes:
                    new_winner = sorted_votes[0][0]
                    final_suspect = new_winner
                    if new_winner == current_winner:
                        winner_stability_counter += 1
                        if winner_stability_counter > 8: is_locked = True
                    else:
                        current_winner = new_winner; winner_stability_counter = 0

        if leak_active and final_suspect:
            detected_events.append({'start': t, 'end': t+1, 'suspect': final_suspect})
            
    return detected_events, prob_history

import tqdm

def clean_id(node_id):
    return str(node_id).strip()

TEST_SCENARIOS= [3, 5, 11, 30, 49, 61, 64, 71, 77, 78, 80, 89, 93, 108, 109, 122, 123, 127, 140, 143, 156, 160, 164, 183, 190, 192, 231, 239, 248, 257, 262, 272, 286, 295, 300, 306, 311, 320, 325, 331, 332, 333, 351, 370, 430, 436, 447, 448, 466, 501, 533, 534, 539, 543, 557, 563, 573, 574, 580, 610, 612, 618, 626, 642, 644, 651, 668, 673, 677, 683, 686, 690, 691, 692, 694, 699, 722, 725, 731, 732, 754, 755, 780, 787, 812, 814, 815, 835, 854, 862, 866, 867, 873, 875, 887, 910, 914, 930, 951, 952, 963, 965, 981, 994, 995]

ALL_TP = 0
ALL_FN = 0
ALL_FP = 0
ALL_TTD = []

print(f"{'Scenario':<10} | {'Status':<15} | {'TTD':<15} | {'GT Node':<10} | {'Found Node':<10}")
print("-" * 75)

for sc_id in tqdm.tqdm(TEST_SCENARIOS):
    ann_df, p_df = load_full_scenario(sc_id, dataset_dir)
    if ann_df is None: continue

    gt_leaks = load_ground_truth(sc_id, dataset_dir)
    ann_norm = (ann_df - mean_stats) / std_stats
    
    detected_raw, _ = run_silent_evaluation(ann_norm, p_df, model, device, gt_leaks)

    if gt_leaks:
        for leak in gt_leaks:
            gt_node = clean_id(leak['node'])
            raw_neighbors = NETWORK_GRAPH.get(gt_node, [])
            valid_set = {gt_node}
            for n in raw_neighbors: valid_set.add(clean_id(n))
            
            start_t = leak['start']
            end_t = leak['end']
            
            active_detections = [d for d in detected_raw if d['start'] >= start_t and d['start'] < end_t]
            first_correct_hit = None
            
            if not active_detections:
                
                ALL_FN += 1
                print(f"{sc_id:<10} | {'MISSED':<15} | {'-':<15} | {gt_node:<10} | {'-':<10}")
            else:
                for d in active_detections:
                    suspect = clean_id(d['suspect'])
                    if suspect in valid_set:
                        first_correct_hit = d
                        break 
                
                if first_correct_hit:
                    ALL_TP += 1
                    steps_taken = first_correct_hit['start'] - start_t
                    mins = steps_taken * 30
                    ALL_TTD.append(mins)
                    
                    ttd_str = f"{int(mins//60)}h {int(mins%60)}m"
                    found_n = clean_id(first_correct_hit['suspect'])
                    print(f"{sc_id:<10} | {'DETECTED':<15} | {ttd_str:<15} | {gt_node:<10} | {found_n:<10}")
                else:
                    ALL_FN += 1
                    bad_suspect = clean_id(active_detections[0]['suspect'])
                    print(f"{sc_id:<10} | {'WRONG NODE':<15} | {'-':<15} | {gt_node:<10} | {bad_suspect:<10}")

    else:
        if detected_raw:
            ALL_FP += 1
            pred = clean_id(detected_raw[0]['suspect'])
            print(f"{sc_id:<10} | {'FALSE ALARM':<15} | {'-':<15} | {'None':<10} | {pred:<10}")

total_leaks = ALL_TP + ALL_FN
precision = ALL_TP / (ALL_TP + ALL_FP) if (ALL_TP + ALL_FP) > 0 else 0.0
recall = ALL_TP / total_leaks if total_leaks > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
avg_ttd_val = np.mean(ALL_TTD) if ALL_TTD else 0.0

print("\n" + "="*45)
print("      FINAL REPORT (PURE ANN)      ")
print("="*45)
print(f"Total Leak Events:      {total_leaks}")
print(f"Clean Scenarios FP:     {ALL_FP}")
print("-" * 45)
print(f"Correct Localizations:  {ALL_TP} (TP)")
print(f"Missed / Wrong Node:    {ALL_FN} (FN)")
print("-" * 45)
print(f"Avg Time to Localize:   {int(avg_ttd_val//60)}h {int(avg_ttd_val%60)}m")
print("-" * 45)
print(f"Precision:              {precision:.4f}")
print(f"Recall:                 {recall:.4f}")
print(f"F1 Score:               {f1:.4f}")
print("="*45)

import matplotlib.pyplot as plt

VIZ_SCENARIO = 320    
ann_df, p_df = load_full_scenario(VIZ_SCENARIO, dataset_dir)

if ann_df is not None:
    ann_norm = (ann_df - mean_stats) / std_stats
    gt_leaks = load_ground_truth(VIZ_SCENARIO, dataset_dir)

    events, probs = run_silent_evaluation(ann_norm, p_df, model, device, gt_leaks)
    
    probs = np.array(probs)
    time_steps = np.arange(len(probs))
    
    is_leak_gt = np.zeros(len(probs), dtype=bool)
    if gt_leaks:
        for leak in gt_leaks:
            is_leak_gt[leak['start'] : leak['end']] = True

    THRESHOLD = 0.5
    is_ml_active = probs >= THRESHOLD

    plt.figure(figsize=(14, 6))
    
    plt.plot(time_steps, probs, color="#2c3e50", linewidth=1.5, label="ANN Confidence")
    plt.axhline(y=THRESHOLD, color='black', linestyle='--', alpha=0.5, label=f"Threshold ({THRESHOLD})")

    plt.fill_between(time_steps, 0, 1.1, where=(is_leak_gt & is_ml_active), 
                     color='green', alpha=0.3, label="Detected (TP)")

    plt.fill_between(time_steps, 0, 1.1, where=(~is_leak_gt & is_ml_active), 
                     color='orange', alpha=0.3, label="False Alarm (FP)")
    
    plt.fill_between(time_steps, 0, 1.1, where=(is_leak_gt & ~is_ml_active), 
                     color='red', alpha=0.3, label="Missed Leak (FN)")

    plt.title(f"Scenario {VIZ_SCENARIO}: Pure ANN Detection Performance", fontsize=14)
    plt.xlabel("Time Steps")
    plt.ylabel("Leak Probability")
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
    print(f"Could not load Scenario {VIZ_SCENARIO}")