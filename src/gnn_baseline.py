
import os
import re
import random
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve
)
from tqdm import tqdm
from datetime import datetime

current_dir = os.getcwd()
Paper_dir = os.path.join(current_dir, "..")
log_dir = os.path.join(Paper_dir, "logs")
result_dir = os.path.join(Paper_dir, "results")
files_dir = os.path.join(Paper_dir, "files")

SCENARIO_DIR = r""
STRUCTURE_DIR = r""
TOPOLOGY_FILE = r"inp_1_text.txt"
MEAN_FILE = os.path.join(files_dir, "mean_gnn_final.txt")
STD_FILE = os.path.join(files_dir, "std_gnn_final.txt")

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

N_NODES = 32
N_PIPES = 34

log_file = open(os.path.join(log_dir, f"log_gnn_{id}.txt"), "a")
log_file.write(f"GNN (GENConv, repo-faithful) with {n_scenarios} scenarios.\n")
log_file.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)

TRAIN_SCENARIOS = []
VAL_SCENARIOS = []
TEST_SCENARIOS = []

log_file.write(f"TRAIN_SCENARIOS: {TRAIN_SCENARIOS}\n")
log_file.write(f"VAL_SCENARIOS: {VAL_SCENARIOS}\n")
log_file.write(f"TEST_SCENARIOS: {TEST_SCENARIOS}\n")
log_file.flush()

def load_scenario_data(scenario_id):
    path = os.path.join(SCENARIO_DIR, f"Scenario-{scenario_id}.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Timestamps" in df.columns:
        df = df.drop(columns=["Timestamps"])
    if "Leaks" not in df.columns:
        raise ValueError(f"Scenario {scenario_id} missing 'Leaks'")
    return df


def add_temporal_columns(df):
    step_of_day = np.arange(len(df)) % 48
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * step_of_day / 48)
    df["cos_hour"] = np.cos(2 * np.pi * step_of_day / 48)
    return df


def load_topology(topology_file):
    edges = []
    with open(topology_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nums = re.findall(r"\d+", line)
            if len(nums) < 3:
                continue
            _, node1, node2 = int(nums[0]), int(nums[1]) - 1, int(nums[2]) - 1
            if 0 <= node1 < N_NODES and 0 <= node2 < N_NODES:
                edges.append((node1, node2))
    if len(edges) != N_PIPES:
        raise ValueError(f"Expected {N_PIPES} pipes, got {len(edges)}")
    edge_pairs = []
    for u, v in edges:
        edge_pairs.append((u, v))
        edge_pairs.append((v, u))
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return edge_index, edges


def get_hydraulic_columns(df):
    excluded = {"Leaks"}
    feature_cols = [c for c in df.columns if c not in excluded]
    hydraulic_cols = [c for c in feature_cols if c not in ["sin_hour", "cos_hour"]]
    if len(hydraulic_cols) < 98:
        raise ValueError(f"Expected >=98 hydraulic cols, found {len(hydraulic_cols)}")
    demand_cols = hydraulic_cols[0:32]
    pressure_cols = hydraulic_cols[32:64]
    flow_cols = hydraulic_cols[64:98]
    return demand_cols, pressure_cols, flow_cols


def compute_and_save_normalization(scenario_ids, mean_file, std_file):
    all_rows = []
    ref_cols = None
    for sid in tqdm(scenario_ids, desc="Computing normalization stats"):
        df = load_scenario_data(sid)
        df = add_temporal_columns(df)
        _ = get_hydraulic_columns(df)
        feature_cols = [c for c in df.columns if c != "Leaks"]
        if ref_cols is None:
            ref_cols = feature_cols
        elif feature_cols != ref_cols:
            raise ValueError(f"Scenario {sid} column layout mismatch")
        all_rows.append(df[feature_cols].values.astype(np.float64))
    stacked = np.concatenate(all_rows, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    os.makedirs(os.path.dirname(mean_file), exist_ok=True)
    np.savetxt(mean_file, mean)
    np.savetxt(std_file, std)
    return mean, std


def normalize_dataframe(df, mean, std):
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "Leaks"]
    df[feature_cols] = (df[feature_cols] - mean) / std
    return df

class WDNGraphDataset(Dataset):
    def __init__(self, scenario_ids, mean, std, window_size=12):
        self.samples = []
        self.window_size = window_size
        for sid in tqdm(scenario_ids, desc="Loading scenarios"):
            df = load_scenario_data(sid)
            df = add_temporal_columns(df)
            demand_cols, pressure_cols, flow_cols = get_hydraulic_columns(df)
            df = normalize_dataframe(df, mean, std)

            demand = df[demand_cols].values.astype(np.float32)
            pressure = df[pressure_cols].values.astype(np.float32)
            flow = df[flow_cols].values.astype(np.float32)
            labels = df["Leaks"].values.astype(np.int64)

            T = len(df)
            for start in range(0, T - window_size + 1):
                end = start + window_size

                node_features = np.zeros((N_NODES, window_size * 2), dtype=np.float32)
                node_features[:, 0:window_size] = demand[start:end].T
                node_features[:, window_size:] = pressure[start:end].T

                pipe_flow = flow[start:end].T.astype(np.float32)  # [N_PIPES, window_size]
                edge_features = np.repeat(pipe_flow, 2, axis=0)   # [2*N_PIPES, window_size]

                target = labels[end - 1]

                self.samples.append((
                    torch.tensor(node_features, dtype=torch.float32),
                    torch.tensor(edge_features, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.long),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_graphs(batch):
    xs, edge_attrs, ys, batch_vector = [], [], [], []
    base_edges = []
    for graph_id, (x, edge_attr, y) in enumerate(batch):
        xs.append(x)
        edge_attrs.append(edge_attr)
        ys.append(y)
        batch_vector.extend([graph_id] * N_NODES)
        base_edges.append(edge_index_global + graph_id * N_NODES)

    x = torch.cat(xs, dim=0)
    edge_attr = torch.cat(edge_attrs, dim=0)
    y = torch.stack(ys)
    batch_vector = torch.tensor(batch_vector, dtype=torch.long)
    batched_edge_index = torch.cat(base_edges, dim=1)
    return x, batched_edge_index, edge_attr, batch_vector, y

class GNNModel(nn.Module):
    def __init__(self, node_in, edge_in, hidden_size=32, target_size=1,
                 heads=1, dropout=0.0, num_layers=4,
                 graph_classification=True, gnn_layer="GENConv", **kwargs):
        super().__init__()
        gnn_layer_cls = getattr(geo_nn, gnn_layer)
        self.graph_classification = graph_classification
        self.node_encoder = nn.Linear(node_in, hidden_size)
        self.num_edge_features = edge_in
        if edge_in:
            self.edge_encoder = nn.Linear(edge_in, hidden_size)
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.num_layers = num_layers

        additional_kw_args = {}
        if edge_in is not None:
            additional_kw_args["edge_dim"] = hidden_size  # edges are pre-encoded to hidden_size

        signature = inspect.signature(gnn_layer_cls)
        if "dropout" in signature.parameters:
            additional_kw_args["dropout"] = dropout
        if "residual" in signature.parameters:
            additional_kw_args["residual"] = True
        if "heads" in signature.parameters:
            additional_kw_args["heads"] = heads

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(gnn_layer_cls(self.hidden_size, self.hidden_size, **additional_kw_args))

        if self.graph_classification:
            self.graph_norm = nn.LayerNorm(self.hidden_size * heads)
            self.graph_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.linear = nn.Linear(self.hidden_size * heads, self.target_size)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        if self.num_edge_features is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, conv in enumerate(self.convs):
            if self.num_edge_features is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, training=self.training)

        if self.graph_classification:
            x = global_mean_pool(x, batch)
            x = self.graph_norm(x)
            x = self.graph_act(x)
            x = self.linear(x)

        return x.squeeze(1)

if not (os.path.exists(MEAN_FILE) and os.path.exists(STD_FILE)):
    print("Normalization files not found — computing from TRAIN_SCENARIOS...")
    mean, std = compute_and_save_normalization(TRAIN_SCENARIOS, MEAN_FILE, STD_FILE)
else:
    print("Found existing normalization files — reusing them.")
    mean = np.loadtxt(MEAN_FILE)
    std = np.loadtxt(STD_FILE)

std = np.where(std == 0, 1.0, std)

edge_index_global, topology_edges = load_topology(TOPOLOGY_FILE)
print(f"Topology loaded: {len(topology_edges)} pipes, {N_NODES} nodes")

train_dataset = WDNGraphDataset(TRAIN_SCENARIOS, mean, std, WINDOW_SIZE)
val_dataset = WDNGraphDataset(VAL_SCENARIOS, mean, std, WINDOW_SIZE)
test_dataset = WDNGraphDataset(TEST_SCENARIOS, mean, std, WINDOW_SIZE)

log_file.write(f"Dataset sizes (graphs): {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}\n")
log_file.flush()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_graphs)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_graphs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_graphs)

train_labels = np.array([s[2].item() for s in train_dataset.samples])
n_pos = int((train_labels == 1).sum())
n_neg = int((train_labels == 0).sum())
weight_for_0 = 1.0 if n_neg == 0 else (n_pos + n_neg) / (2.0 * n_neg)
weight_for_1 = 1.0 if n_pos == 0 else (n_pos + n_neg) / (2.0 * n_pos)
pos_weight = torch.tensor(weight_for_1 / weight_for_0, device=DEVICE)

print(f"Class weights (approx): neg={weight_for_0:.3f}, pos={weight_for_1:.3f}")
log_file.write(f"Train class counts: pos={n_pos}, neg={n_neg}\n")
log_file.write(f"Class weights (approx): neg={weight_for_0:.3f}, pos={weight_for_1:.3f}\n")
log_file.flush()

NODE_IN = WINDOW_SIZE * 2
EDGE_IN = WINDOW_SIZE

model = GNNModel(
    node_in=NODE_IN,
    edge_in=EDGE_IN,
    hidden_size=64,
    target_size=1,
    heads=1,
    dropout=DROPOUT,
    num_layers=4,
    graph_classification=True,
    gnn_layer="GENConv",
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    preds_all, labs_all = [], []
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for xb, edge_idx, edge_attr, batch_vec, yb in tqdm(loader, disable=not train_mode):
            xb = xb.to(DEVICE)
            edge_idx = edge_idx.to(DEVICE)
            edge_attr = edge_attr.to(DEVICE)
            batch_vec = batch_vec.to(DEVICE)
            yb = yb.float().to(DEVICE)

            logits = model(xb, edge_idx, edge_attr, batch_vec)
            loss = criterion(logits, yb)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * xb_graphs(batch_vec)
            preds_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            labs_all.append(yb.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labs_all = np.concatenate(labs_all)
    pred_labels = (preds_all >= 0.5).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average="binary", zero_division=0)
    avg_loss = total_loss / len(preds_all)
    return avg_loss, f1


def xb_graphs(batch_vec):
    return int(batch_vec.max().item()) + 1

best_val_f1 = -1.0
ckpt_path = os.path.join(files_dir, f"_gnn_last_{id}_{n_scenarios}_{WINDOW_SIZE}.pth")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_f1 = run_epoch(train_loader, train_mode=True)
    val_loss, val_f1 = run_epoch(val_loader, train_mode=False)

    print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f} | train_f1: {train_f1:.6f} "
          f"| val_loss: {val_loss:.6f} | val_f1: {val_f1:.4f}")
    log_file.write(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f} | train_f1: {train_f1:.6f} "
                    f"| val_loss: {val_loss:.6f} | val_f1: {val_f1:.4f}\n")
    log_file.flush()

    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model at epoch {epoch}")

print("Training finished. Best Val F1:", best_val_f1)


model.load_state_dict(torch.load(ckpt_path))
model.eval()

preds_all, labs_all = [], []
with torch.no_grad():
    for xb, edge_idx, edge_attr, batch_vec, yb in tqdm(test_loader):
        xb = xb.to(DEVICE)
        edge_idx = edge_idx.to(DEVICE)
        edge_attr = edge_attr.to(DEVICE)
        batch_vec = batch_vec.to(DEVICE)
        logits = model(xb, edge_idx, edge_attr, batch_vec)
        preds_all.append(torch.sigmoid(logits).cpu().numpy())
        labs_all.append(yb.numpy())

preds_all = np.concatenate(preds_all)
labs_all = np.concatenate(labs_all)

fpr, tpr, thresholds = roc_curve(labs_all, preds_all)
roc_auc = roc_auc_score(labs_all, preds_all)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

log_file.write(f"Best Threshold: {best_threshold}\n")
print("Best threshold =", best_threshold)
print("ROC-AUC =", roc_auc)

pred_labels = (preds_all >= best_threshold).astype(int)
acc = accuracy_score(labs_all, pred_labels)
p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average="binary", zero_division=0)
cm = confusion_matrix(labs_all, pred_labels)

print(f"Test metrics: acc={acc:.4f}, precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}")
print("Confusion matrix:\n", cm)

log_file.write(f"Test metrics: acc={acc:.4f}, precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}\n")
log_file.write(f"Confusion matrix: {cm.tolist()}\n")
log_file.flush()
log_file.close()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")