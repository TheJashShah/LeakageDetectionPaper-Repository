
import os

current_dir = os.getcwd()
workspace = os.path.join(current_dir, "..")
log_dir = os.path.join(workspace, "logs")
result_dir = os.path.join(workspace, "results")
files_dir = os.path.join(workspace, "files")

from datetime import datetime
id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
n_scenarios = 700

log_file = open(os.path.join(log_dir, f"log_ann_physics_{id}.txt"), "a")
log_file.write(f"ANN with Physics with {n_scenarios} scenarios. New Approach.\n")
log_file.flush()

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

def load_scenario_data(scenario_id):

    data  = pd.read_csv(os.path.join(workspace, f"Scenario-{scenario_id}.csv"))
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

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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

TRAIN_SCENARIOS = [1, 4, 8, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 35, 37, 42, 44, 46, 48, 50, 51, 52, 53, 55, 56, 57, 59, 63, 67, 68, 69, 70, 72, 74, 75, 82, 83, 84, 86, 88, 92, 95, 96, 97, 98, 99, 101, 105, 106, 110, 113, 115, 116, 117, 120, 125, 128, 129, 130, 134, 135, 136, 137, 138, 141, 145, 146, 148, 150, 152, 153, 155, 157, 162, 165, 168, 171, 172, 173, 174, 176, 178, 180, 184, 185, 186, 187, 188, 193, 194, 198, 199, 200, 201, 202, 204, 205, 207, 208, 210, 211, 212, 216, 218, 219, 224, 226, 227, 228, 229, 230, 233, 234, 236, 238, 241, 242, 244, 245, 246, 247, 250, 252, 253, 254, 255, 256, 258, 260, 261, 263, 264, 268, 271, 273, 274, 276, 277, 278, 279, 280, 282, 285, 288, 290, 291, 296, 299, 301, 308, 309, 313, 314, 316, 318, 319, 324, 328, 329, 334, 336, 337, 341, 342, 343, 344, 345, 346, 347, 349, 354, 362, 363, 364, 366, 367, 371, 372, 376, 378, 381, 382, 383, 384, 387, 391, 392, 393, 398, 399, 400, 402, 403, 404, 405, 406, 408, 409, 411, 412, 416, 417, 418, 421, 422, 423, 424, 425, 426, 431, 432, 435, 437, 438, 439, 441, 445, 450, 451, 452, 453, 455, 456, 457, 459, 460, 461, 462, 463, 464, 465, 470, 471, 474, 475, 477, 478, 480, 483, 488, 490, 493, 494, 495, 497, 498, 502, 507, 508, 509, 511, 512, 514, 515, 516, 517, 519, 521, 524, 530, 536, 537, 540, 541, 542, 544, 545, 549, 551, 552, 553, 554, 555, 556, 559, 560, 565, 567, 570, 571, 572, 575, 576, 578, 579, 582, 583, 585, 588, 590, 591, 592, 593, 597, 598, 601, 603, 604, 605, 606, 609, 616, 620, 621, 622, 623, 625, 629, 630, 631, 632, 635, 636, 637, 638, 640, 641, 643, 646, 647, 650, 652, 657, 658, 659, 660, 663, 664, 665, 666, 670, 672, 674, 675, 676, 678, 681, 682, 684, 685, 687, 688, 696, 697, 701, 705, 707, 709, 713, 714, 715, 716, 717, 718, 720, 724, 727, 735, 736, 738, 740, 743, 745, 747, 749, 752, 757, 759, 760, 761, 762, 763, 766, 767, 768, 769, 771, 772, 775, 777, 779, 781, 782, 785, 786, 788, 789, 790, 794, 795, 798, 799, 802, 805, 806, 807, 808, 810, 811, 817, 819, 820, 821, 822, 823, 824, 825, 828, 830, 831, 832, 836, 839, 840, 841, 842, 843, 844, 849, 851, 852, 855, 860, 861, 863, 864, 865, 868, 874, 876, 877, 878, 881, 882, 883, 884, 886, 890, 891, 892, 893, 895, 900, 901, 903, 904, 907, 908, 909, 911, 913, 916, 919, 920, 922, 923, 924, 925, 926, 927, 935, 937, 939, 940, 941, 948, 955, 956, 957, 958, 959, 961, 967, 968, 969, 970, 973, 974, 977, 979, 983, 985, 986, 988, 991, 992, 997]
VAL_SCENARIOS = [7, 26, 27, 28, 38, 40, 43, 45, 47, 54, 65, 66, 79, 102, 124, 149, 154, 158, 161, 163, 167, 177, 179, 203, 206, 214, 235, 249, 265, 270, 302, 315, 321, 326, 338, 339, 348, 356, 358, 361, 373, 375, 377, 388, 389, 394, 395, 413, 414, 427, 433, 434, 442, 443, 446, 458, 472, 482, 485, 500, 518, 522, 527, 532, 535, 564, 566, 586, 587, 595, 613, 615, 617, 627, 649, 653, 656, 679, 710, 721, 728, 733, 737, 741, 748, 753, 773, 792, 800, 804, 827, 846, 859, 879, 889, 898, 899, 905, 928, 929, 936, 943, 954, 975, 978]
TEST_SCENARIOS = [3, 5, 11, 30, 49, 61, 64, 71, 77, 78, 80, 89, 93, 108, 109, 122, 123, 127, 140, 143, 156, 160, 164, 183, 190, 192, 231, 239, 248, 257, 262, 272, 286, 295, 300, 306, 311, 320, 325, 331, 332, 333, 351, 370, 430, 436, 447, 448, 466, 501, 533, 534, 539, 543, 557, 563, 573, 574, 580, 610, 612, 618, 626, 642, 644, 651, 668, 673, 677, 683, 686, 690, 691, 692, 694, 699, 722, 725, 731, 732, 754, 755, 780, 787, 812, 814, 815, 835, 854, 862, 866, 867, 873, 875, 887, 910, 914, 930, 951, 952, 963, 965, 981, 994, 995]

log_file.write(f"TRAIN_SCENARIOS: {TRAIN_SCENARIOS}\n")
log_file.write(f"VAL_SCENARIOS: {VAL_SCENARIOS}\n")
log_file.write(f"TEST_SCENARIOS: {TEST_SCENARIOS}\n")
log_file.flush()

def load_edge_index(path, device):
    
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment=";",
        header=None,
        names=["ID", "Node1", "Node2"]
    )
    
    src = torch.tensor(df["Node1"].values - 1, dtype=torch.long, device=device)
    dst = torch.tensor(df["Node2"].values - 1, dtype=torch.long, device=device)
    
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index

import os

def load_pipe_params(scenario_id, device):
    
    base_path = workspace
    csv_path = os.path.join(base_path, f"Scenario-{scenario_id}-structure.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        L = torch.tensor(df["length"].values, dtype=torch.float32, device=device)
        D = torch.tensor(df["diameter"].values, dtype=torch.float32, device=device)
        C = torch.tensor(df["roughness"].values, dtype=torch.float32, device=device)
    
        return L, D, C
    
    else:
        return None

def load_all_pipe_params(scenario_ids):
    
    param_dict = {}
    
    for sid in scenario_ids:
        params = load_pipe_params(sid, device=DEVICE)
        
        if params is None:
            continue
        
        L, D, C = params
        param_dict[sid] = (L, D, C)
    
    return param_dict

def physics_informed_loss(
    leak_loss,
    pred_p, pred_f,
    true_d,
    L, D, C,
    edge_index,
    lambda_mass=0.5,
    lambda_energy=0.5,
    lambda_phys=0.05
):
    B, n_pipes = pred_f.shape
    n_nodes = pred_p.shape[1]
    
    src = edge_index[0]
    dst = edge_index[1]
    
    flow_out = torch.zeros((B, n_nodes), device=DEVICE)
    flow_in = torch.zeros((B, n_nodes), device=DEVICE)
    
    flow_out.index_add_(1, src, pred_f)
    flow_in.index_add_(1, dst, pred_f)
    
    mass_residual = flow_in - flow_out - true_d
    mass_scale = torch.mean(torch.abs(true_d)) + 1e-6
    mass_loss = torch.mean((mass_residual / mass_scale) ** 2)

    q = torch.abs(pred_f) + 1e-6
    
    head_loss = (
        10.67 * L* (q ** 1.852) / ((C ** 1.852) * (D ** 4.87) + 1e-6)     
    )
    
    sign = torch.sign(pred_f)
    pred_drop = sign * (pred_p[:, src] - pred_p[:, dst])
    energy_scale = torch.mean(torch.abs(pred_p)) + 1e-6
    energy_residual = pred_drop - head_loss
    energy_loss = torch.mean((energy_residual / energy_scale) ** 2)
    
    phys_loss = lambda_mass * mass_loss + lambda_energy * energy_loss
    total_loss = leak_loss + lambda_phys * phys_loss
    
    return total_loss, mass_loss.detach(), energy_loss.detach(), phys_loss

def main():
    
    PIPE_PARAMS = load_all_pipe_params(
        TRAIN_SCENARIOS + VAL_SCENARIOS + TEST_SCENARIOS
    )
    
    edge_index = load_edge_index(path=os.path.join(workspace, "inp_1_text.txt"), device=DEVICE)
    
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
    np.savetxt(os.path.join(files_dir, "mean_ann_physics.txt"), mean)
    np.savetxt(os.path.join(files_dir, "std_ann_physics.txt"), std)
    
    demand_mean = mean[0:32]
    pressure_mean = mean[32:64]
    flow_mean = mean[64:98]
    
    demand_std = std[0:32]
    pressure_std = std[32:64]
    flow_std = std[64:98]
    
    np.savez(
        os.path.join(files_dir, "scalers_ann_physics.npz"),
        demand_mean=demand_mean,
        demand_std=demand_std,
        press_mean=pressure_mean,
        press_std=pressure_std,
        flow_mean=flow_mean,
        flow_std=flow_std
    )
    
    scalers = np.load(os.path.join(files_dir, "scalers_ann_physics.npz"))
    demand_mean = torch.tensor(scalers["demand_mean"], device=DEVICE)
    demand_std  = torch.tensor(scalers["demand_std"], device=DEVICE)

    press_mean  = torch.tensor(scalers["press_mean"], device=DEVICE)
    press_std   = torch.tensor(scalers["press_std"], device=DEVICE)

    flow_mean   = torch.tensor(scalers["flow_mean"], device=DEVICE)
    flow_std    = torch.tensor(scalers["flow_std"], device=DEVICE)

    train_dfs = [normalize_df(df, mean, std) for df in train_dfs]
    val_dfs = [normalize_df(df, mean, std) for df in val_dfs]
    test_dfs = [normalize_df(df, mean, std) for df in test_dfs]
    
    train_ds = WindowedScenarioDataset(train_dfs, scenario_ids=TRAIN_SCENARIOS, window_size=WINDOW_SIZE, step=1)
    val_ds = WindowedScenarioDataset(val_dfs, scenario_ids=VAL_SCENARIOS, window_size=WINDOW_SIZE, step=1)
    test_ds = WindowedScenarioDataset(test_dfs, scenario_ids=TEST_SCENARIOS, window_size=WINDOW_SIZE, step=1)
    print("Dataset sizes (windows):", len(train_ds), len(val_ds), len(test_ds))
    log_file.write(f"Dataset sizes (windows):, {len(train_ds)}, {len(val_ds)}, {len(test_ds)}\n")
    
    input_dim = train_ds.X.shape[1]
    print(f"Input Dimension: {input_dim}")
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
    
    model  = LeakPhysicsANN(input_dim).to(DEVICE)
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
        "val_f1": [],
        "train_physics_loss": [],
    }
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_phys_loss = 0.0
        
        if epoch < 10:
            lambda_phys = 0.01
        elif epoch < 20:
            lambda_phys = 0.03
        else:
            lambda_phys = 0.05
            
        for xb, yb, sid_batch, true_d in tqdm.tqdm(train_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            
            L_batch = torch.stack([PIPE_PARAMS[int(sid)][0] for sid in sid_batch])
            D_batch = torch.stack([PIPE_PARAMS[int(sid)][1] for sid in sid_batch])
            C_batch = torch.stack([PIPE_PARAMS[int(sid)][2] for sid in sid_batch])
            
            true_d = true_d.to(DEVICE)
            true_d = true_d * demand_std + demand_mean
            
            leak_prob, pred_p, pred_f = model(xb)
            
            pred_p = pred_p * press_std + press_mean
            pred_f = pred_f * flow_std + flow_mean
            
            leak_loss = bce_weighted(leak_prob, yb)
            
            loss, _, _, phys_loss = physics_informed_loss(
                leak_loss,
                pred_p,
                pred_f,
                true_d,
                L_batch,
                D_batch,
                C_batch,
                edge_index,
                lambda_mass=1.0,
                lambda_energy=0.1,
                lambda_phys=lambda_phys
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * (xb.size(0))
            running_phys_loss += phys_loss.item() *(xb.size(0)) 
            
        train_loss = running_loss / len(train_ds)
        phy_loss = running_phys_loss / len(train_ds)
        history["train_loss"].append(train_loss)
        history["train_physics_loss"].append(phy_loss)
        
        model.eval()
        val_loss = 0.0
        train_preds_all = []
        train_labs_all = []
        val_preds_all = []
        val_labs_all = []
        
        with torch.no_grad():
            for xb, yb, sid_batch, true_d in tqdm.tqdm(val_loader):
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                
                L_batch = torch.stack([PIPE_PARAMS[int(sid)][0] for sid in sid_batch])
                D_batch = torch.stack([PIPE_PARAMS[int(sid)][1] for sid in sid_batch])
                C_batch = torch.stack([PIPE_PARAMS[int(sid)][2] for sid in sid_batch])
                
                true_d = true_d.to(DEVICE)
                true_d = true_d * demand_std + demand_mean
                
                leak_prob, pred_p, pred_f = model(xb)
                
                pred_p = pred_p * press_std + press_mean
                pred_f = pred_f * flow_std + flow_mean
            
                leak_loss = bce_weighted(leak_prob, yb)
                
                loss, _, _, phys_loss = physics_informed_loss(
                    leak_loss,
                    pred_p,
                    pred_f,
                    true_d,
                    L_batch,
                    D_batch,
                    C_batch,
                    edge_index,
                    lambda_mass=1.0,
                    lambda_energy=0.1,
                    lambda_phys=lambda_phys
                )

                val_loss += loss.item() * xb.size(0)
                val_preds_all.append(leak_prob.cpu().numpy())
                val_labs_all.append(yb.cpu().numpy())
                
        val_loss = val_loss / len(val_ds)
        val_preds_all = np.concatenate(val_preds_all)
        val_labs_all = np.concatenate(val_labs_all)
        val_pred_labels = (val_preds_all >= 0.5).astype(int)
        p, r, val_f1, _ = precision_recall_fscore_support(val_labs_all, val_pred_labels, average='binary', zero_division=0)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | val_f1: {val_f1:.4f}")
        log_file.write(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | val_f1: {val_f1:.4f}\n")
        
        if epoch >= 3:
            scheduler.step(val_f1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        torch.save(model.state_dict(), os.path.join(files_dir, f"ann_{epoch}_physics_{id}.pth"))
        print(f"Model Saved for Epoch {epoch}.")
    
    torch.save(model.state_dict(), os.path.join(files_dir, f"_ann_last_{id}_physics.pth"))
    print(f"Saved Model.")
            
    print("Training Finished. Best Val F1: ", best_val_f1)
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history["val_f1"], label="val_f1")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.legend()
    plt.title("F1")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"training_curves_{id}_ann_physics.png"))
    
    plt.figure(figsize=(6,4))
    plt.plot(history["train_physics_loss"], label="physics_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Physics Loss")
    plt.title("Physics Loss")
    plt.legend()

    plt.savefig(os.path.join(result_dir, f"physics_loss_ann_{id}.png"))
    plt.show()
    
    model.load_state_dict(torch.load(os.path.join(files_dir, f"_ann_last_{id}_physics.pth")))
    model.eval()
    
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
    plt.savefig(os.path.join(result_dir, f"roc_curve_test_{id}_ann_physics.png"))
    plt.show()
    
    pred_labels = (preds_all >= best_threshold).astype(int)

    acc = accuracy_score(labs_all, pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average='binary', zero_division=0)
    cm = confusion_matrix(labs_all, pred_labels)

    print("Test metrics: acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".format(acc, p, r, f1))
    print("Confusion matrix:\n", cm)
    
    log_file.write(f"Test metrics: acc = {acc:.4f}, precision = {p:.4f}, recall = {r:.4f}, f1 = {f1:.4f}")
    log_file.write(f"Confusion matrix: {cm}")

    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix (test)")
    plt.colorbar()

    plt.xlabel("Predicted (Pred 0 / Pred 1)")
    plt.ylabel("True (True 0 / True 1)")
        
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center', 
                    color='white' if cm[i,j] > cm.max()/2 else 'black')

    plt.savefig(os.path.join(result_dir, f"confusion_matrix_test_{id}_ann_physics.png"))
    plt.show()
    
    pred_labels_05 = (preds_all >= 0.5).astype(int)

    acc_05 = accuracy_score(labs_all, pred_labels_05)
    p_05, r_05, f1_05, _ = precision_recall_fscore_support(labs_all, pred_labels_05, average='binary', zero_division=0)
    cm_05 = confusion_matrix(labs_all, pred_labels_05)

    print("Test metrics for threshold 0.5: acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".format(acc_05, p_05, r_05, f1_05))
    print("Confusion matrix for threshold 0.5:\n", cm_05)
    
    log_file.write(f"Test metrics for threshold 0.5: acc = {acc_05:.4f}, precision = {p_05:.4f}, recall = {r_05:.4f}, f1 = {f1_05:.4f}")
    log_file.write(f"Confusion matrix for threshold 0.5: {cm_05}")

    plt.figure(figsize=(4,4))
    plt.imshow(cm_05, interpolation='nearest')
    plt.title("Confusion matrix (test)")
    plt.colorbar()

    plt.xlabel("Predicted (Pred 0 / Pred 1)")
    plt.ylabel("True (True 0 / True 1)")
        
    for i in range(cm_05.shape[0]):
        for j in range(cm_05.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center', 
                    color='white' if cm[i,j] > cm.max()/2 else 'black')

    plt.savefig(os.path.join(result_dir, f"confusion_matrix_test_{id}_ann_physics_base_threshold.png"))
    plt.show()
    
    log_file.flush()
    log_file.close()
    
main()

