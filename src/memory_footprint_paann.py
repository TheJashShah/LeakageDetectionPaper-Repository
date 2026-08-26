 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)

import tqdm
import os
import time
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


 
def load_scenario_data(scenario_id):

    data = pd.read_csv(
        os.path.join(
            r"",
            f"Scenario-{scenario_id}.csv"
        )
    )

    data = data.drop(columns=["Unnamed: 0"])

    return data


 
def add_temporal_columns(df):

    step_of_day = np.array([
        i % 48
        for i in range(len(df))
    ])

    df["sin_hour"] = np.sin(
        2 * np.pi * step_of_day / 48
    )

    df["cos_hour"] = np.cos(
        2 * np.pi * step_of_day / 48
    )

    return df


 
WINDOW_SIZE = 12
BATCH_SIZE = 256

LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 30
DROPOUT = 0.5

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


 
class WindowedScenarioDataset(Dataset):

    def __init__(
        self,
        df_list,
        scenario_ids,
        window_size=5,
        step=1
    ):

        self.window_size = window_size
        self.step = step

        self.X = []
        self.y = []
        self.scenario_ids = []
        self.demands = []

        for df, sid in zip(
            df_list,
            scenario_ids
        ):

            arr = df.drop(
                columns=["Leaks"]
            ).values.astype(np.float32)

            labs = df["Leaks"].values.astype(
                np.int64
            )

            T, F = arr.shape

            for start in range(
                0,
                T - window_size + 1,
                step
            ):

                window = arr[
                    start:start + window_size
                ].reshape(-1)

                target = labs[
                    start + window_size - 1
                ]

                self.X.append(window)
                self.y.append(target)

                self.scenario_ids.append(sid)

                self.demands.append(
                    arr[
                        start + window_size - 1,
                        0:32
                    ]
                )

        self.X = np.array(
            self.X,
            dtype=np.float32
        )

        self.y = np.array(
            self.y,
            dtype=np.int64
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx],
            self.scenario_ids[idx],
            torch.tensor(
                self.demands[idx]
            )
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

            nn.Linear(
                input_dim,
                hidden1
            ),

            nn.BatchNorm1d(
                hidden1
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            )
        )

        self.layer2 = nn.Sequential(

            nn.Linear(
                hidden1,
                hidden2
            ),

            nn.BatchNorm1d(
                hidden2
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            )
        )

        self.layer3 = nn.Sequential(

            nn.Linear(
                hidden2,
                hidden3
            ),

            nn.BatchNorm1d(
                hidden3
            ),

            nn.ReLU()
        )

        # Hydraulic prediction heads
        self.pressure_head = nn.Linear(
            hidden3,
            n_nodes
        )

        self.flow_head = nn.Linear(
            hidden3,
            n_pipes
        )

        # Classification head
        self.leak_head = nn.Sequential(

            nn.Linear(
                hidden3 + 5,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        h = self.layer1(x)

        h = self.layer2(h)

        h = self.layer3(h)

        pred_p = self.pressure_head(h)

        pred_f = self.flow_head(h)

        # Physics-derived features
        p_mean = pred_p.mean(
            dim=1
        )

        p_std = pred_p.std(
            dim=1
        )

        p_rng = (
            pred_p.max(
                dim=1
            ).values
            -
            pred_p.min(
                dim=1
            ).values
        )

        f_mean = pred_f.mean(
            dim=1
        )

        f_std = pred_f.std(
            dim=1
        )

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
            [
                h,
                phys_feat
            ],
            dim=1
        )

        leak_prob = torch.sigmoid(
            self.leak_head(
                leak_input
            )
        ).squeeze(1)

        return (
            leak_prob,
            pred_p,
            pred_f
        )


 
def normalize_df(
    df,
    mean,
    std
):

    cols = [
        c for c in df.columns
        if c != "Leaks"
    ]

    df2 = df.copy()

    df2[cols] = (
        df2[cols] - mean
    ) / std

    return df2


 
TEST_SCENARIOS = []


 
MEAN_PATH = ()

STD_PATH = ()

MODEL_PATH = ()
 
def load_test_data():

    test_dfs = []

    print("Loading test scenarios...")

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
                    f"Scenario {sid} "
                    f"missing 'Leaks' column"
                )

            df = add_temporal_columns(
                df
            )

            test_dfs.append(df)

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

    test_dfs = [
        normalize_df(
            df,
            mean,
            std
        )
        for df in test_dfs
    ]

    return test_dfs


 
def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


 
def parameter_memory_mb(model):

    total_bytes = sum(
        p.numel() *
        p.element_size()
        for p in model.parameters()
    )

    return total_bytes / (
        1024 ** 2
    )


 
def measure_inference(
    model,
    test_loader
):

    model.eval()

    # --------------------------------------------------
    # GPU warm-up
    # --------------------------------------------------

    print(
        "\nRunning GPU warm-up..."
    )

    warmup_batches = 10

    with torch.no_grad():

        for i, batch in enumerate(
            test_loader
        ):

            xb = batch[0]

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            _ = model(xb)

            if i + 1 >= warmup_batches:
                break

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # Timed inference
    # --------------------------------------------------

    total_samples = 0
    total_batches = 0

    batch_times = []

    # IMPORTANT:
    # Do NOT store all predictions.
    # This keeps the memory measurement clean.

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    all_scores = []
    all_labels = []

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    overall_start = time.perf_counter()

    with torch.no_grad():

        for batch in tqdm.tqdm(
            test_loader,
            desc="Inference"
        ):

            xb = batch[0]
            yb = batch[1]

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            # Synchronize before timing
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            # FULL PA-ANN forward pass
            #
            # This executes:
            #   hidden layers
            #   pressure head
            #   flow head
            #   physics features
            #   classification head
            #
            leak_prob, pred_p, pred_f = model(xb)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            batch_time = (
                end - start
            )

            batch_times.append(
                batch_time
            )

            # Classification evaluation
            pred_labels = (
                leak_prob >= 0.5
            ).long()

            y_cpu = yb.cpu()
            pred_cpu = pred_labels.cpu()
            prob_cpu = leak_prob.cpu()

            # Confusion matrix counts
            tp += int(
                (
                    (pred_cpu == 1) &
                    (y_cpu == 1)
                ).sum()
            )

            tn += int(
                (
                    (pred_cpu == 0) &
                    (y_cpu == 0)
                ).sum()
            )

            fp += int(
                (
                    (pred_cpu == 1) &
                    (y_cpu == 0)
                ).sum()
            )

            fn += int(
                (
                    (pred_cpu == 0) &
                    (y_cpu == 1)
                ).sum()
            )

            # ROC-AUC requires scores
            all_scores.append(
                prob_cpu.numpy()
            )

            all_labels.append(
                y_cpu.numpy()
            )

            total_samples += (
                xb.size(0)
            )

            total_batches += 1

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    overall_end = time.perf_counter()

    model_inference_time = (
        overall_end -
        overall_start
    )

    mean_batch_latency = np.mean(
        batch_times
    )

    median_batch_latency = np.median(
        batch_times
    )

    std_batch_latency = np.std(
        batch_times
    )

    latency_per_sample = (
        model_inference_time /
        total_samples
    )

    throughput = (
        total_samples /
        model_inference_time
    )

    # --------------------------------------------------
    # Peak GPU memory
    # --------------------------------------------------

    if DEVICE.type == "cuda":

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        peak_gpu_reserved_mb = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        )

    else:

        peak_gpu_memory_mb = None
        peak_gpu_reserved_mb = None

    return {
        "total_samples": total_samples,
        "total_batches": total_batches,

        "model_inference_time":
            model_inference_time,

        "mean_batch_latency":
            mean_batch_latency,

        "median_batch_latency":
            median_batch_latency,

        "std_batch_latency":
            std_batch_latency,

        "latency_per_sample":
            latency_per_sample,

        "throughput":
            throughput,

        "peak_gpu_memory_mb":
            peak_gpu_memory_mb,

        "peak_gpu_reserved_mb":
            peak_gpu_reserved_mb,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "scores":
            np.concatenate(
                all_scores
            ),

        "labels":
            np.concatenate(
                all_labels
            )
    }


 
def main():

    print("=" * 70)

    print(
        "FINAL PA-ANN INFERENCE "
        "TIME / MEMORY / LATENCY EVALUATION"
    )

    print("=" * 70)

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    test_dfs = load_test_data()

    print(
        f"\nScenarios evaluated: "
        f"{len(test_dfs)}/{len(TEST_SCENARIOS)}"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    test_ds = WindowedScenarioDataset(
        test_dfs,
        scenario_ids=TEST_SCENARIOS,
        window_size=WINDOW_SIZE,
        step=1
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    print(
        f"Total inference samples: "
        f"{len(test_ds):,}"
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = LeakPhysicsANN(
        1200
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    # --------------------------------------------------
    # Model size
    # --------------------------------------------------

    n_params = count_parameters(
        model
    )

    parameter_mb = parameter_memory_mb(
        model
    )

    model_file_size_mb = (
        os.path.getsize(
            MODEL_PATH
        )
        / (1024 ** 2)
    )

    # --------------------------------------------------
    # CPU memory
    # --------------------------------------------------

    if PSUTIL_AVAILABLE:

        process = psutil.Process(
            os.getpid()
        )

        cpu_memory_before_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

    else:

        cpu_memory_before_mb = None

    # --------------------------------------------------
    # Inference benchmark
    # --------------------------------------------------

    results = measure_inference(
        model,
        test_loader
    )

    # --------------------------------------------------
    # CPU memory after
    # --------------------------------------------------

    if PSUTIL_AVAILABLE:

        cpu_memory_after_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        cpu_memory_change_mb = (
            cpu_memory_after_mb -
            cpu_memory_before_mb
        )

    else:

        cpu_memory_after_mb = None
        cpu_memory_change_mb = None

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    tp = results["tp"]
    tn = results["tn"]
    fp = results["fp"]
    fn = results["fn"]

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) /
        total
    )

    precision = (
        tp /
        (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp /
        (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    roc_auc = roc_auc_score(
        results["labels"],
        results["scores"]
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "CLASSIFICATION"
    )
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall   : {recall:.6f}"
    )

    print(
        f"F1       : {f1:.6f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.6f}"
    )

    print("\nConfusion matrix:")

    print(
        np.array([
            [tn, fp],
            [fn, tp]
        ])
    )

    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "MODEL SIZE"
    )
    print("=" * 70)

    print(
        f"Trainable parameters : "
        f"{n_params:,}"
    )

    print(
        f"Parameter memory     : "
        f"{parameter_mb:.3f} MB"
    )

    print(
        f"Checkpoint file size : "
        f"{model_file_size_mb:.3f} MB"
    )

    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "INFERENCE TIME / LATENCY"
    )
    print("=" * 70)

    print(
        f"Total samples              : "
        f"{results['total_samples']:,}"
    )

    print(
        f"Total batches              : "
        f"{results['total_batches']:,}"
    )

    print(
        f"Total model inference time : "
        f"{results['model_inference_time']:.6f} s"
    )

    print(
        f"Mean batch latency         : "
        f"{results['mean_batch_latency'] * 1000:.4f} ms"
    )

    print(
        f"Median batch latency       : "
        f"{results['median_batch_latency'] * 1000:.4f} ms"
    )

    print(
        f"Std batch latency          : "
        f"{results['std_batch_latency'] * 1000:.4f} ms"
    )

    print(
        f"Latency per sample         : "
        f"{results['latency_per_sample'] * 1000:.6f} ms"
    )

    print(
        f"Throughput                : "
        f"{results['throughput']:.2f} samples/s"
    )

    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "MEMORY FOOTPRINT"
    )
    print("=" * 70)

    if (
        results["peak_gpu_memory_mb"]
        is not None
    ):

        print(
            f"Peak GPU allocated memory : "
            f"{results['peak_gpu_memory_mb']:.3f} MB"
        )

        print(
            f"Peak GPU reserved memory  : "
            f"{results['peak_gpu_reserved_mb']:.3f} MB"
        )

    else:

        print(
            "GPU memory: N/A "
            "(CPU inference)"
        )

    if PSUTIL_AVAILABLE:

        print(
            f"CPU RSS before inference  : "
            f"{cpu_memory_before_mb:.3f} MB"
        )

        print(
            f"CPU RSS after inference   : "
            f"{cpu_memory_after_mb:.3f} MB"
        )

        print(
            f"CPU RSS change             : "
            f"{cpu_memory_change_mb:.3f} MB"
        )

    else:

        print(
            "\nCPU RSS measurement "
            "requires psutil."
        )

    print("\n")
    print("=" * 70)
    print(
        "END OF PA-ANN EVALUATION"
    )
    print("=" * 70)


 
main()