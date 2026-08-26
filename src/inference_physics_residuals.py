
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

import matplotlib.pyplot as plt
import tqdm
import os
import json

def load_scenario_data(scenario_id):

    data = pd.read_csv(
        os.path.join(
            r"D:\scenario_df",
            f"Scenario-{scenario_id}.csv"
        )
    )

    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    return data

def add_temporal_columns(df):

    step_of_day = np.array(
        [i % 48 for i in range(len(df))]
    )

    df["sin_hour"] = np.sin(
        2 * np.pi * step_of_day / 48
    )

    df["cos_hour"] = np.cos(
        2 * np.pi * step_of_day / 48
    )

    return df

 
WINDOW_SIZE = 12
BATCH_SIZE = 256

DROPOUT = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_NODES = 32
N_PIPES = 34

DEMAND_SLICE = slice(0, 32)
PRESSURE_SLICE = slice(32, 64)
FLOW_SLICE = slice(64, 98)

INPUT_DIM = WINDOW_SIZE * 100
# 32 demand + 32 pressure + 34 flow + 2 temporal = 100
 
TEST_SCENARIOS = []

CHECKPOINT_PATH = ()

MEAN_PATH = ()

STD_PATH = ()

PIPE_TOPOLOGY_PATH = ()

RESULTS_DIR = ()

SAVE_RAW = False

 
class WindowedScenarioDatasetPhysics(Dataset):


    def __init__(
        self,
        df_list,
        scenario_ids,
        window_size=12,
        step=1
    ):

        self.window_size = window_size
        self.step = step

        self.scenarios = []
        self.index_map = []

        for df, sid in zip(df_list, scenario_ids):

            arr = (
                df.drop(columns=["Leaks"])
                .values
                .astype(np.float32)
            )

            labs = (
                df["Leaks"]
                .values
                .astype(np.int64)
            )

            T = len(arr)

            n_windows = (
                (T - window_size) // step
            ) + 1

            if n_windows <= 0:
                continue

            scenario_index = len(self.scenarios)

            self.scenarios.append(
                {
                    "sid": int(sid),
                    "arr": arr,
                    "labs": labs,
                    "n_windows": n_windows
                }
            )

            for window_idx in range(n_windows):
                self.index_map.append(
                    (
                        scenario_index,
                        window_idx
                    )
                )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        scenario_index, window_idx = self.index_map[idx]

        scenario = self.scenarios[scenario_index]

        arr = scenario["arr"]
        labs = scenario["labs"]
        sid = scenario["sid"]

        start = window_idx * self.step
        target_idx = start + self.window_size - 1

        window = arr[
            start:start + self.window_size
        ].reshape(-1)

        target_row = arr[target_idx]

        label = labs[target_idx]

        demand = target_row[DEMAND_SLICE]
        true_p = target_row[PRESSURE_SLICE]
        true_f = target_row[FLOW_SLICE]

        return (
            torch.from_numpy(window),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(sid, dtype=torch.long),
            torch.from_numpy(demand),
            torch.from_numpy(true_p),
            torch.from_numpy(true_f)
        )
 
class LeakPhysicsANN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1=512,
        hidden2=256,
        hidden3=64,
        n_nodes=32,
        n_pipes=34
    ):

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

        # Predicted pressure at 32 nodes
        self.pressure_head = nn.Linear(
            hidden3,
            n_nodes
        )

        # Predicted flow at 34 pipes
        self.flow_head = nn.Linear(
            hidden3,
            n_pipes
        )

        # Final classification head
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

        # Physics-derived features
        p_mean = pred_p.mean(dim=1)
        p_std = pred_p.std(dim=1)

        p_rng = (
            pred_p.max(dim=1).values
            - pred_p.min(dim=1).values
        )

        f_mean = pred_f.mean(dim=1)
        f_std = pred_f.std(dim=1)

        phys_feat = torch.stack(
            [
                p_mean,
                p_std,
                p_rng,
                f_mean,
                f_std
            ],
            dim=1
        )

        leak_input = torch.cat(
            [h, phys_feat],
            dim=1
        )

        leak_prob = torch.sigmoid(
            self.leak_head(leak_input)
        ).squeeze(1)

        return (
            leak_prob,
            pred_p,
            pred_f
        )

def normalize_df(df, mean, std):

    cols = [
        c for c in df.columns
        if c != "Leaks"
    ]

    df2 = df.copy()

    df2[cols] = (
        df2[cols] - mean
    ) / std

    return df2


EDGE_INDEX_PATH = r"inp_1_text.txt"

STRUCTURE_BASE_PATH = r""

RESULTS_DIR = r""

def load_edge_index(path, device):

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment=";",
        header=None,
        names=["ID", "Node1", "Node2"]
    )

    src = torch.tensor(
        df["Node1"].values - 1,
        dtype=torch.long,
        device=device
    )

    dst = torch.tensor(
        df["Node2"].values - 1,
        dtype=torch.long,
        device=device
    )

    edge_index = torch.stack(
        [src, dst],
        dim=0
    )

    if edge_index.shape[1] != N_PIPES:
        raise ValueError(
            f"Expected {N_PIPES} pipes in edge file, "
            f"found {edge_index.shape[1]}"
        )

    return edge_index

def load_pipe_params(scenario_id, device):

    csv_path = os.path.join(
        STRUCTURE_BASE_PATH,
        f"Scenario-{scenario_id}-structure.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Structure file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "length",
        "diameter",
        "roughness"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Scenario {scenario_id} structure file is missing "
            f"columns: {missing}"
        )

    if len(df) != N_PIPES:
        raise ValueError(
            f"Scenario {scenario_id}: expected "
            f"{N_PIPES} pipes, found {len(df)}"
        )

    L = torch.tensor(
        df["length"].values,
        dtype=torch.float32,
        device=device
    )

    D = torch.tensor(
        df["diameter"].values,
        dtype=torch.float32,
        device=device
    )

    C = torch.tensor(
        df["roughness"].values,
        dtype=torch.float32,
        device=device
    )

    return L, D, C

class ChannelRegressionAccumulator:

    def __init__(self, n_channels):

        self.n_channels = n_channels

        self.n = np.zeros(
            n_channels,
            dtype=np.float64
        )

        self.sum_y = np.zeros(
            n_channels,
            dtype=np.float64
        )

        self.sum_y2 = np.zeros(
            n_channels,
            dtype=np.float64
        )

        self.sum_sq_err = np.zeros(
            n_channels,
            dtype=np.float64
        )

    def update(self, y_true, y_pred):

        mask = (
            np.isfinite(y_true)
            &
            np.isfinite(y_pred)
        )

        yt = np.where(
            mask,
            y_true,
            0.0
        )

        err2 = np.where(
            mask,
            (y_true - y_pred) ** 2,
            0.0
        )

        self.n += mask.sum(axis=0)

        self.sum_y += yt.sum(axis=0)

        self.sum_y2 += (
            yt ** 2
        ).sum(axis=0)

        self.sum_sq_err += (
            err2
        ).sum(axis=0)

    def finalize(self):

        n_safe = np.maximum(
            self.n,
            1
        )

        mse = (
            self.sum_sq_err
            /
            n_safe
        )

        rmse = np.sqrt(mse)

        mean_y = (
            self.sum_y
            /
            n_safe
        )

        ss_tot = (
            self.sum_y2
            -
            self.n * mean_y ** 2
        )

        r2 = np.where(
            ss_tot > 1e-8,
            1.0 -
            (
                self.sum_sq_err
                /
                np.maximum(
                    ss_tot,
                    1e-8
                )
            ),
            np.nan
        )

        return mse, rmse, r2

    def global_finalize(self):

        n = self.n.sum()

        sse = self.sum_sq_err.sum()

        sy = self.sum_y.sum()

        sy2 = self.sum_y2.sum()

        n_safe = max(
            n,
            1
        )

        mse = (
            sse
            /
            n_safe
        )

        rmse = np.sqrt(mse)

        mean_y = (
            sy
            /
            n_safe
        )

        ss_tot = (
            sy2
            -
            n * mean_y ** 2
        )

        if ss_tot > 1e-8:

            r2 = (
                1.0
                -
                sse / ss_tot
            )

        else:

            r2 = np.nan

        return (
            mse,
            rmse,
            r2,
            n
        )

class ResidualAccumulator:

    def __init__(self):

        self.n = 0.0

        self.sum_r = 0.0

        self.sum_abs = 0.0

        self.sum_r2 = 0.0

    def update(self, residual):

        residual = np.asarray(
            residual
        )

        mask = np.isfinite(
            residual
        )

        r = residual[mask]

        self.n += r.size

        if r.size == 0:
            return

        self.sum_r += r.sum()

        self.sum_abs += np.abs(
            r
        ).sum()

        self.sum_r2 += (
            r ** 2
        ).sum()

    def finalize(self):

        if self.n == 0:

            return {
                "mean": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "n": 0
            }

        mean = (
            self.sum_r
            /
            self.n
        )

        mae = (
            self.sum_abs
            /
            self.n
        )

        mse = (
            self.sum_r2
            /
            self.n
        )

        rmse = np.sqrt(
            mse
        )

        return {
            "mean": float(mean),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "n": int(self.n)
        }

def compute_mass_residual(
    pred_f,
    demand,
    edge_index
):

    src = edge_index[0]

    dst = edge_index[1]

    B = pred_f.shape[0]

    n_nodes = N_NODES

    flow_in = torch.zeros(
        B,
        n_nodes,
        device=pred_f.device,
        dtype=pred_f.dtype
    )

    flow_out = torch.zeros(
        B,
        n_nodes,
        device=pred_f.device,
        dtype=pred_f.dtype
    )

    flow_in.index_add_(
        1,
        dst,
        pred_f
    )

    flow_out.index_add_(
        1,
        src,
        pred_f
    )

    residual = (
        flow_in
        -
        flow_out
        -
        demand
    )

    return residual


def compute_energy_residual(
    pred_p,
    pred_f,
    L,
    D,
    C,
    edge_index
):

    src = edge_index[0]

    dst = edge_index[1]

    Q = pred_f

    p_src = pred_p[:, src]

    p_dst = pred_p[:, dst]

    delta_p = (
        torch.sign(Q)
        *
        (p_src - p_dst)
    )

    # Same numerical protection used in training
    Q_abs = (
        torch.abs(Q)
        +
        1e-6
    )

    h = (
        10.67
        *
        L
        *
        Q_abs.pow(1.852)
        /
        (
            C.pow(1.852)
            *
            D.pow(4.87)
        )
    )

    residual = (
        delta_p
        -
        h
    )

    return residual

def evaluate_physical_consistency(
    results_dir=RESULTS_DIR
):

    os.makedirs(
        results_dir,
        exist_ok=True
    )



    print("\nLoading WDN topology...")

    edge_index = load_edge_index(
        EDGE_INDEX_PATH,
        DEVICE
    )

    print(
        f"Topology loaded: "
        f"{edge_index.shape[1]} pipes"
    )


    test_dfs = []

    valid_scenarios = []

    print("\nLoading test scenarios...")

    for sid in tqdm.tqdm(
        TEST_SCENARIOS
    ):

        try:

            df = load_scenario_data(
                sid
            )

            if df is None or len(df) == 0:
                continue

            if "Timestamps" in df.columns:

                df = df.drop(
                    columns=["Timestamps"]
                )

            if "Leaks" not in df.columns:

                raise ValueError(
                    f"Scenario {sid} missing "
                    f"'Leaks' column"
                )

            df = add_temporal_columns(
                df
            )

            # Verify structure file exists
            structure_path = os.path.join(
                STRUCTURE_BASE_PATH,
                f"Scenario-{sid}-structure.csv"
            )

            if not os.path.exists(
                structure_path
            ):

                raise FileNotFoundError(
                    f"Missing structure file: "
                    f"{structure_path}"
                )

            test_dfs.append(df)

            valid_scenarios.append(
                sid
            )

        except Exception as e:

            print(
                f"Skipping scenario {sid}: "
                f"{repr(e)}"
            )

    mean = np.loadtxt(
        MEAN_PATH
    )

    std = np.loadtxt(
        STD_PATH
    )

    test_dfs_norm = [
        normalize_df(
            df,
            mean,
            std
        )
        for df in test_dfs
    ]


    dataset = WindowedScenarioDatasetPhysics(
        test_dfs_norm,
        scenario_ids=valid_scenarios,
        window_size=WINDOW_SIZE,
        step=1
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"\nScenarios evaluated: "
        f"{len(valid_scenarios)}/{len(TEST_SCENARIOS)}"
    )

    print(
        f"Total inference rows: "
        f"{len(dataset):,}"
    )


    model = LeakPhysicsANN(
        1200
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    mean_t = torch.tensor(
        mean,
        dtype=torch.float32,
        device=DEVICE
    )

    std_t = torch.tensor(
        std,
        dtype=torch.float32,
        device=DEVICE
    )

    p_mean = mean_t[
        32:64
    ]

    p_std = std_t[
        32:64
    ]

    f_mean = mean_t[
        64:98
    ]

    f_std = std_t[
        64:98
    ]

    d_mean = mean_t[
        0:32
    ]

    d_std = std_t[
        0:32
    ]

    pressure_acc = ChannelRegressionAccumulator(
        N_NODES
    )

    flow_acc = ChannelRegressionAccumulator(
        N_PIPES
    )

    # Predicted-state residuals
    mass_pred_acc = ResidualAccumulator()
    energy_pred_acc = ResidualAccumulator()

    # True-state residuals
    mass_true_acc = ResidualAccumulator()
    energy_true_acc = ResidualAccumulator()

    # True-state NON-LEAK
    mass_true_nonleak_acc = ResidualAccumulator()
    energy_true_nonleak_acc = ResidualAccumulator()

    # True-state LEAK
    mass_true_leak_acc = ResidualAccumulator()
    energy_true_leak_acc = ResidualAccumulator()


    preds_all = []
    labs_all = []

    per_scenario = {}


    def get_scenario_acc(
        sid
    ):

        if sid not in per_scenario:

            per_scenario[sid] = {

                "pressure_acc":
                    ChannelRegressionAccumulator(
                        N_NODES
                    ),

                "flow_acc":
                    ChannelRegressionAccumulator(
                        N_PIPES
                    ),

                "mass_pred":
                    ResidualAccumulator(),

                "energy_pred":
                    ResidualAccumulator(),

                "mass_true":
                    ResidualAccumulator(),

                "energy_true":
                    ResidualAccumulator(),

                "mass_true_nonleak":
                    ResidualAccumulator(),

                "energy_true_nonleak":
                    ResidualAccumulator(),

                "mass_true_leak":
                    ResidualAccumulator(),

                "energy_true_leak":
                    ResidualAccumulator(),

                "n_rows": 0,

                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0
            }

        return per_scenario[sid]

    with torch.no_grad():

        for batch in tqdm.tqdm(
            loader,
            desc="Physical evaluation"
        ):

            (
                xb,
                yb,
                sid_batch,
                demand_n,
                true_p_n,
                true_f_n
            ) = batch


            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            demand_n = demand_n.to(
                DEVICE,
                non_blocking=True
            )

            true_p_n = true_p_n.to(
                DEVICE,
                non_blocking=True
            )

            true_f_n = true_f_n.to(
                DEVICE,
                non_blocking=True
            )


            leak_prob, pred_p_n, pred_f_n = model(
                xb
            )

            pred_labels = (
                leak_prob >= 0.5
            ).long()

            preds_all.append(
                leak_prob.cpu().numpy()
            )

            labs_all.append(
                yb.numpy()
            )

            pred_p = (
                pred_p_n
                *
                p_std
                +
                p_mean
            )

            pred_f = (
                pred_f_n
                *
                f_std
                +
                f_mean
            )

            true_p = (
                true_p_n
                *
                p_std
                +
                p_mean
            )

            true_f = (
                true_f_n
                *
                f_std
                +
                f_mean
            )

            demand = (
                demand_n
                *
                d_std
                +
                d_mean
            )


            pred_p_np = (
                pred_p
                .cpu()
                .numpy()
            )

            true_p_np = (
                true_p
                .cpu()
                .numpy()
            )

            pred_f_np = (
                pred_f
                .cpu()
                .numpy()
            )

            true_f_np = (
                true_f
                .cpu()
                .numpy()
            )


            pressure_acc.update(
                true_p_np,
                pred_p_np
            )

            flow_acc.update(
                true_f_np,
                pred_f_np
            )

            sid_np = np.asarray(
                sid_batch
            )

            unique_sids = np.unique(
                sid_np
            )

            for sid in unique_sids:

                sid = int(sid)

                mask = (
                    sid_np == sid
                )

                L, D, C = load_pipe_params(
                    sid,
                    DEVICE
                )

                pred_p_sid = pred_p[
                    mask
                ]

                pred_f_sid = pred_f[
                    mask
                ]

                true_p_sid = true_p[
                    mask
                ]

                true_f_sid = true_f[
                    mask
                ]

                demand_sid = demand[
                    mask
                ]

                mass_pred = compute_mass_residual(
                    pred_f_sid,
                    demand_sid,
                    edge_index
                )

                mass_true = compute_mass_residual(
                    true_f_sid,
                    demand_sid,
                    edge_index
                )

                energy_pred = compute_energy_residual(
                    pred_p_sid,
                    pred_f_sid,
                    L,
                    D,
                    C,
                    edge_index
                )

                energy_true = compute_energy_residual(
                    true_p_sid,
                    true_f_sid,
                    L,
                    D,
                    C,
                    edge_index
                )

                mass_pred_np = (
                    mass_pred
                    .cpu()
                    .numpy()
                )

                mass_true_np = (
                    mass_true
                    .cpu()
                    .numpy()
                )

                energy_pred_np = (
                    energy_pred
                    .cpu()
                    .numpy()
                )

                energy_true_np = (
                    energy_true
                    .cpu()
                    .numpy()
                )


                y_true_sid = (
                    yb.numpy()[mask]
                )

                nonleak_mask = (
                    y_true_sid == 0
                )

                leak_mask = (
                    y_true_sid == 1
                )

                mass_pred_acc.update(
                    mass_pred_np.flatten()
                )

                energy_pred_acc.update(
                    energy_pred_np.flatten()
                )


                mass_true_acc.update(
                    mass_true_np.flatten()
                )

                energy_true_acc.update(
                    energy_true_np.flatten()
                )

                if nonleak_mask.any():

                    mass_true_nonleak_acc.update(
                        mass_true_np[
                            nonleak_mask
                        ].flatten()
                    )

                    energy_true_nonleak_acc.update(
                        energy_true_np[
                            nonleak_mask
                        ].flatten()
                    )


                if leak_mask.any():

                    mass_true_leak_acc.update(
                        mass_true_np[
                            leak_mask
                        ].flatten()
                    )

                    energy_true_leak_acc.update(
                        energy_true_np[
                            leak_mask
                        ].flatten()
                    )

                acc = get_scenario_acc(
                    sid
                )

                acc["n_rows"] += int(
                    mask.sum()
                )


                # Pressure / flow
                acc["pressure_acc"].update(
                    true_p_np[mask],
                    pred_p_np[mask]
                )

                acc["flow_acc"].update(
                    true_f_np[mask],
                    pred_f_np[mask]
                )


                # Predicted residual
                acc["mass_pred"].update(
                    mass_pred_np.flatten()
                )

                acc["energy_pred"].update(
                    energy_pred_np.flatten()
                )


                # True residual
                acc["mass_true"].update(
                    mass_true_np.flatten()
                )

                acc["energy_true"].update(
                    energy_true_np.flatten()
                )


                # True non-leak
                if nonleak_mask.any():

                    acc["mass_true_nonleak"].update(
                        mass_true_np[
                            nonleak_mask
                        ].flatten()
                    )

                    acc["energy_true_nonleak"].update(
                        energy_true_np[
                            nonleak_mask
                        ].flatten()
                    )


                # True leak
                if leak_mask.any():

                    acc["mass_true_leak"].update(
                        mass_true_np[
                            leak_mask
                        ].flatten()
                    )

                    acc["energy_true_leak"].update(
                        energy_true_np[
                            leak_mask
                        ].flatten()
                    )


                yp = (
                    pred_labels
                    .cpu()
                    .numpy()[mask]
                )

                yt = y_true_sid

                acc["tp"] += int(
                    (
                        (yt == 1)
                        &
                        (yp == 1)
                    ).sum()
                )

                acc["fp"] += int(
                    (
                        (yt == 0)
                        &
                        (yp == 1)
                    ).sum()
                )

                acc["tn"] += int(
                    (
                        (yt == 0)
                        &
                        (yp == 0)
                    ).sum()
                )

                acc["fn"] += int(
                    (
                        (yt == 1)
                        &
                        (yp == 0)
                    ).sum()
                )


            del xb
            del demand_n
            del true_p_n
            del true_f_n

            del leak_prob
            del pred_p_n
            del pred_f_n

            del pred_p
            del pred_f
            del true_p
            del true_f
            del demand

            if DEVICE.type == "cuda":

                torch.cuda.empty_cache()



    preds_all = np.concatenate(
        preds_all
    )

    labs_all = np.concatenate(
        labs_all
    )

    pred_labels_all = (
        preds_all >= 0.5
    ).astype(int)


    classification_acc = accuracy_score(
        labs_all,
        pred_labels_all
    )

    classification_precision, \
    classification_recall, \
    classification_f1, _ = (
        precision_recall_fscore_support(
            labs_all,
            pred_labels_all,
            average="binary",
            zero_division=0
        )
    )

    cm = confusion_matrix(
        labs_all,
        pred_labels_all
    )

    pressure_mse, \
    pressure_rmse, \
    pressure_r2, \
    pressure_n = pressure_acc.global_finalize()


    flow_mse, \
    flow_rmse, \
    flow_r2, \
    flow_n = flow_acc.global_finalize()

    mass_pred = (
        mass_pred_acc.finalize()
    )

    energy_pred = (
        energy_pred_acc.finalize()
    )

    mass_true = (
        mass_true_acc.finalize()
    )

    energy_true = (
        energy_true_acc.finalize()
    )

    mass_true_nonleak = (
        mass_true_nonleak_acc.finalize()
    )

    energy_true_nonleak = (
        energy_true_nonleak_acc.finalize()
    )

    mass_true_leak = (
        mass_true_leak_acc.finalize()
    )

    energy_true_leak = (
        energy_true_leak_acc.finalize()
    )



    pressure_mse_ch, \
    pressure_rmse_ch, \
    pressure_r2_ch = (
        pressure_acc.finalize()
    )

    pd.DataFrame({

        "node_idx":
            np.arange(N_NODES),

        "mse":
            pressure_mse_ch,

        "rmse":
            pressure_rmse_ch,

        "r2":
            pressure_r2_ch

    }).to_csv(
        os.path.join(
            results_dir,
            "pressure_channel_metrics.csv"
        ),
        index=False
    )

    flow_mse_ch, \
    flow_rmse_ch, \
    flow_r2_ch = (
        flow_acc.finalize()
    )

    pd.DataFrame({

        "pipe_idx":
            np.arange(N_PIPES),

        "mse":
            flow_mse_ch,

        "rmse":
            flow_rmse_ch,

        "r2":
            flow_r2_ch

    }).to_csv(
        os.path.join(
            results_dir,
            "flow_channel_metrics.csv"
        ),
        index=False
    )

    scenario_rows = []


    for sid, acc in sorted(
        per_scenario.items()
    ):



        pmse, prmse, pr2, pn = (
            acc[
                "pressure_acc"
            ].global_finalize()
        )


        fmse, frmse, fr2, fn = (
            acc[
                "flow_acc"
            ].global_finalize()
        )

        mp = acc[
            "mass_pred"
        ].finalize()

        ep = acc[
            "energy_pred"
        ].finalize()

        mt = acc[
            "mass_true"
        ].finalize()

        et = acc[
            "energy_true"
        ].finalize()

        mtn = acc[
            "mass_true_nonleak"
        ].finalize()

        etn = acc[
            "energy_true_nonleak"
        ].finalize()

        mtl = acc[
            "mass_true_leak"
        ].finalize()

        etl = acc[
            "energy_true_leak"
        ].finalize()


        tp = acc["tp"]
        fp = acc["fp"]
        tn = acc["tn"]
        fn = acc["fn"]

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else np.nan
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else np.nan
        )

        if (
            np.isfinite(precision)
            and
            np.isfinite(recall)
            and
            precision + recall > 0
        ):

            f1 = (
                2
                *
                precision
                *
                recall
                /
                (
                    precision
                    +
                    recall
                )
            )

        else:

            f1 = np.nan


        total = (
            tp
            +
            fp
            +
            tn
            +
            fn
        )

        accuracy = (
            (tp + tn) / total
            if total > 0
            else np.nan
        )


        scenario_rows.append({

            "scenario_id":
                sid,

            "n_rows":
                acc["n_rows"],

            # Classification
            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            # Hydraulic prediction
            "pressure_mse":
                pmse,

            "pressure_rmse":
                prmse,

            "pressure_r2":
                pr2,

            "flow_mse":
                fmse,

            "flow_rmse":
                frmse,

            "flow_r2":
                fr2,

            # Predicted-state mass
            "mass_pred_mae":
                mp["mae"],

            "mass_pred_mse":
                mp["mse"],

            "mass_pred_rmse":
                mp["rmse"],

            # True-state mass
            "mass_true_mae":
                mt["mae"],

            "mass_true_mse":
                mt["mse"],

            "mass_true_rmse":
                mt["rmse"],

            # True non-leak mass
            "mass_true_nonleak_mae":
                mtn["mae"],

            "mass_true_nonleak_mse":
                mtn["mse"],

            "mass_true_nonleak_rmse":
                mtn["rmse"],

            # True leak mass
            "mass_true_leak_mae":
                mtl["mae"],

            "mass_true_leak_mse":
                mtl["mse"],

            "mass_true_leak_rmse":
                mtl["rmse"],

            # Predicted-state energy
            "energy_pred_mae":
                ep["mae"],

            "energy_pred_mse":
                ep["mse"],

            "energy_pred_rmse":
                ep["rmse"],

            # True-state energy
            "energy_true_mae":
                et["mae"],

            "energy_true_mse":
                et["mse"],

            "energy_true_rmse":
                et["rmse"],

            # True non-leak energy
            "energy_true_nonleak_mae":
                etn["mae"],

            "energy_true_nonleak_mse":
                etn["mse"],

            "energy_true_nonleak_rmse":
                etn["rmse"],

            # True leak energy
            "energy_true_leak_mae":
                etl["mae"],

            "energy_true_leak_mse":
                etl["mse"],

            "energy_true_leak_rmse":
                etl["rmse"]
        })


    scenario_df = pd.DataFrame(
        scenario_rows
    )

    scenario_df.to_csv(
        os.path.join(
            results_dir,
            "per_scenario_physics_metrics.csv"
        ),
        index=False
    )


    overall = {

        "n_scenarios_evaluated":
            len(valid_scenarios),

        "n_scenarios_requested":
            len(TEST_SCENARIOS),

        "n_valid_inference_rows":
            int(len(labs_all)),


        "classification": {

            "accuracy":
                float(
                    classification_acc
                ),

            "precision":
                float(
                    classification_precision
                ),

            "recall":
                float(
                    classification_recall
                ),

            "f1":
                float(
                    classification_f1
                )
        },

        "pressure_prediction": {

            "mse":
                float(
                    pressure_mse
                ),

            "rmse":
                float(
                    pressure_rmse
                ),

            "r2":
                float(
                    pressure_r2
                ),

            "values_evaluated":
                int(
                    pressure_n
                )
        },

        "flow_prediction": {

            "mse":
                float(
                    flow_mse
                ),

            "rmse":
                float(
                    flow_rmse
                ),

            "r2":
                float(
                    flow_r2
                ),

            "values_evaluated":
                int(
                    flow_n
                )
        },


        "mass_residual": {

            "predicted_state":
                mass_pred,

            "true_state":
                mass_true,

            "true_nonleak":
                mass_true_nonleak,

            "true_leak":
                mass_true_leak
        },

        "energy_residual": {

            "predicted_state":
                energy_pred,

            "true_state":
                energy_true,

            "true_nonleak":
                energy_true_nonleak,

            "true_leak":
                energy_true_leak
        }
    }


    with open(
        os.path.join(
            results_dir,
            "overall_physics_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            overall,
            f,
            indent=2
        )


    print("\n")
    print("=" * 75)
    print("FINAL UNSEEN-DATA PHYSICAL CONSISTENCY EVALUATION")
    print("=" * 75)

    print(
        f"\nScenarios evaluated: "
        f"{len(valid_scenarios)}/{len(TEST_SCENARIOS)}"
    )

    print("\n")
    print("POOLED CLASSIFICATION")
    print("-" * 50)

    print(
        f"Accuracy : "
        f"{classification_acc:.6f}"
    )

    print(
        f"Precision: "
        f"{classification_precision:.6f}"
    )

    print(
        f"Recall   : "
        f"{classification_recall:.6f}"
    )

    print(
        f"F1       : "
        f"{classification_f1:.6f}"
    )

    print("\n")
    print("POOLED PRESSURE PREDICTION")
    print("-" * 50)

    print(
        f"MSE : "
        f"{pressure_mse:.6f}"
    )

    print(
        f"RMSE: "
        f"{pressure_rmse:.6f}"
    )

    print(
        f"R2  : "
        f"{pressure_r2:.6f}"
    )

    print(
        f"Values evaluated: "
        f"{int(pressure_n):,}"
    )


    print("\n")
    print("POOLED FLOW PREDICTION")
    print("-" * 50)

    print(
        f"MSE : "
        f"{flow_mse:.6f}"
    )

    print(
        f"RMSE: "
        f"{flow_rmse:.6f}"
    )

    print(
        f"R2  : "
        f"{flow_r2:.6f}"
    )

    print(
        f"Values evaluated: "
        f"{int(flow_n):,}"
    )


    print("\n")
    print("MASS CONSERVATION RESIDUAL")
    print("-" * 50)

    print("\nPredicted state:")
    print(
        f"MAE : "
        f"{mass_pred['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{mass_pred['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{mass_pred['rmse']:.6f}"
    )


    print("\nTrue state:")
    print(
        f"MAE : "
        f"{mass_true['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{mass_true['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{mass_true['rmse']:.6f}"
    )


    print("\nTrue state — NON-LEAK:")
    print(
        f"MAE : "
        f"{mass_true_nonleak['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{mass_true_nonleak['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{mass_true_nonleak['rmse']:.6f}"
    )


    print("\nTrue state — LEAK:")
    print(
        f"MAE : "
        f"{mass_true_leak['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{mass_true_leak['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{mass_true_leak['rmse']:.6f}"
    )



    print("\n")
    print("ENERGY CONSERVATION RESIDUAL")
    print("-" * 50)

    print("\nPredicted state:")
    print(
        f"MAE : "
        f"{energy_pred['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{energy_pred['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{energy_pred['rmse']:.6f}"
    )


    print("\nTrue state:")
    print(
        f"MAE : "
        f"{energy_true['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{energy_true['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{energy_true['rmse']:.6f}"
    )


    print("\nTrue state — NON-LEAK:")
    print(
        f"MAE : "
        f"{energy_true_nonleak['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{energy_true_nonleak['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{energy_true_nonleak['rmse']:.6f}"
    )


    print("\nTrue state — LEAK:")
    print(
        f"MAE : "
        f"{energy_true_leak['mae']:.6f}"
    )
    print(
        f"MSE : "
        f"{energy_true_leak['mse']:.6f}"
    )
    print(
        f"RMSE: "
        f"{energy_true_leak['rmse']:.6f}"
    )

    print("\n")
    print("CONFUSION MATRIX")
    print("-" * 50)

    print(cm)

    print("\n")
    print("RESULTS SAVED TO:")
    print(
        os.path.abspath(
            results_dir
        )
    )

    print("\nFiles:")
    print(
        "  pressure_channel_metrics.csv"
    )
    print(
        "  flow_channel_metrics.csv"
    )
    print(
        "  per_scenario_physics_metrics.csv"
    )
    print(
        "  overall_physics_metrics.json"
    )

    print("=" * 75)

    return overall

overall_results = evaluate_physical_consistency()