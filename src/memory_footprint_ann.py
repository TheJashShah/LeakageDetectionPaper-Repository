 
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
import matplotlib.pyplot as plt
import random
import tqdm
import os
import time
import gc

# Optional: CPU memory measurement
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

    step_of_day = np.array([i % 48 for i in range(len(df))])

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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


 
class WindowedScenarioDataset(Dataset):

    def __init__(
        self,
        df_list,
        window_size=5,
        step=1
    ):

        self.window_size = window_size
        self.step = step

        self.X = []
        self.y = []

        for df in df_list:

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

        return self.X[idx], self.y[idx]


 
class LeakANN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1=512,
        hidden2=256,
        hidden3=64
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

        self.out = nn.Linear(
            hidden3,
            1
        )

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        return torch.sigmoid(
            self.out(x)
        ).squeeze(1)


 
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
# PATHS

MEAN_PATH = r""

STD_PATH = r""

MODEL_PATH = r""

 
def load_test_data():

    test_dfs = []

    print("Loading test scenarios...")

    for sid in tqdm.tqdm(
        TEST_SCENARIOS
    ):

        try:

            df = load_scenario_data(sid)

            if df is None or len(df) == 0:
                continue

            if "Timestamps" in df.columns:

                df = df.drop(
                    columns=["Timestamps"]
                )

            if "Leaks" not in df.columns:

                raise ValueError(
                    f"Scenario {sid} missing 'Leaks' column"
                )

            df = add_temporal_columns(df)

            test_dfs.append(df)

        except Exception as e:

            print(
                f"Skipping scenario {sid}: {repr(e)}"
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
        p.numel() * p.element_size()
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
    # Warm-up
    # --------------------------------------------------

    print("\nRunning GPU warm-up...")

    warmup_batches = 10

    with torch.no_grad():

        for i, (xb, yb) in enumerate(
            test_loader
        ):

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

    all_preds = []
    all_labels = []

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    overall_start = time.perf_counter()

    with torch.no_grad():

        for xb, yb in tqdm.tqdm(
            test_loader,
            desc="Inference"
        ):

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------
            # Batch latency
            # ------------------------------

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            preds = model(xb)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            batch_time = end - start

            batch_times.append(
                batch_time
            )

            # ------------------------------

            total_samples += xb.size(0)
            total_batches += 1

            all_preds.append(
                preds.cpu().numpy()
            )

            all_labels.append(
                yb.cpu().numpy()
            )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    overall_end = time.perf_counter()

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

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

    preds_all = np.concatenate(
        all_preds
    )

    labs_all = np.concatenate(
        all_labels
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
        "preds": preds_all,
        "labels": labs_all,
        "total_samples": total_samples,
        "total_batches": total_batches,
        "model_inference_time": model_inference_time,
        "mean_batch_latency": mean_batch_latency,
        "median_batch_latency": median_batch_latency,
        "std_batch_latency": std_batch_latency,
        "latency_per_sample": latency_per_sample,
        "throughput": throughput,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "peak_gpu_reserved_mb": peak_gpu_reserved_mb
    }


 
def evaluate_classification(
    preds_all,
    labs_all
):

    pred_labels = (
        preds_all >= 0.5
    ).astype(int)

    acc = accuracy_score(
        labs_all,
        pred_labels
    )

    p, r, f1, _ = (
        precision_recall_fscore_support(
            labs_all,
            pred_labels,
            average="binary",
            zero_division=0
        )
    )

    cm = confusion_matrix(
        labs_all,
        pred_labels
    )

    roc_auc = roc_auc_score(
        labs_all,
        preds_all
    )

    return (
        acc,
        p,
        r,
        f1,
        roc_auc,
        cm
    )


 
def main():

    print("=" * 70)

    print(
        "FINAL ANN INFERENCE "
        "TIME / MEMORY / LATENCY EVALUATION"
    )

    print("=" * 70)

    # --------------------------------------------------
    # Load data
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
        window_size=WINDOW_SIZE,
        step=1
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda")
    )

    print(
        f"Total inference samples: "
        f"{len(test_ds):,}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = LeakANN(
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
    # CPU memory before inference
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
    # Inference
    # --------------------------------------------------

    results = measure_inference(
        model,
        test_loader
    )

    # --------------------------------------------------
    # CPU memory after inference
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

    acc, p, r, f1, roc_auc, cm = (
        evaluate_classification(
            results["preds"],
            results["labels"]
        )
    )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "CLASSIFICATION"
    )
    print("=" * 70)

    print(
        f"Accuracy : {acc:.6f}"
    )

    print(
        f"Precision: {p:.6f}"
    )

    print(
        f"Recall   : {r:.6f}"
    )

    print(
        f"F1       : {f1:.6f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.6f}"
    )

    print("\nConfusion matrix:")
    print(cm)

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
        f"Total model inference time: "
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
        f"Latency per sample        : "
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

    if results["peak_gpu_memory_mb"] is not None:

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
            "(inference performed on CPU)"
        )

    if PSUTIL_AVAILABLE:

        print(
            f"CPU RSS before inference : "
            f"{cpu_memory_before_mb:.3f} MB"
        )

        print(
            f"CPU RSS after inference  : "
            f"{cpu_memory_after_mb:.3f} MB"
        )

        print(
            f"CPU RSS change            : "
            f"{cpu_memory_change_mb:.3f} MB"
        )

    else:

        print(
            "\nCPU memory measurement "
            "requires: pip install psutil"
        )

    print("\n")
    print("=" * 70)
    print(
        "END OF EVALUATION"
    )
    print("=" * 70)


 
main()