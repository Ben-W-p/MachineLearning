#!/usr/bin/env python3
"""
pycuda_knn_dataset_runner.py

Reusable CPU-vs-GPU brute-force KNN benchmark.

What it does
------------
1) CPU baseline: traditional nested loops over test and training samples.
2) GPU version: computes pairwise squared distances on the GPU with PyCUDA.
3) Both paths use the same KNN voting step so predictions can be compared.

Data sources
------------
- Built-in datasets from scikit-learn:
  iris, wbc (breast cancer), wine, digits, mnist_784, covtype, synthetic
- Any CSV file with numeric feature columns and a target column

Examples
--------
# Small built-in dataset
python pycuda_knn_dataset_runner.py --dataset iris

# Wisconsin breast cancer dataset
python pycuda_knn_dataset_runner.py --dataset wbc --k 5

# Digits dataset
python pycuda_knn_dataset_runner.py --dataset digits --k 3

# Large dataset from scikit-learn (downloaded on first use)
python pycuda_knn_dataset_runner.py --dataset covtype --limit-train 20000 --limit-test 2000

# MNIST from OpenML (downloaded on first use)
python pycuda_knn_dataset_runner.py --dataset mnist_784 --limit-train 10000 --limit-test 1000

# Your own CSV, using the last column as the target
python pycuda_knn_dataset_runner.py --csv ./my_data.csv

# Your own CSV, specifying the target column name
python pycuda_knn_dataset_runner.py --csv ./wbc.csv --target-column diagnosis

Notes
-----
- CSV features must be numeric. String labels are okay; they are label-encoded.
- The benchmark processes the test set in batches so larger datasets can fit in memory.
- CPU brute-force KNN on very large datasets may still take a long time. Use
  --limit-train / --limit-test when you want a manageable comparison.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyCUDA is not installed or CUDA is not configured correctly.\n"
        "Install with: pip install pycuda\n"
        "Also make sure the CUDA toolkit and driver are installed and nvcc is on PATH."
    ) from exc


@dataclass
class BenchmarkResult:
    name: str
    predictions: np.ndarray
    elapsed_s: float
    extra: dict


CUDA_SRC = r"""
extern "C" __global__ void pairwise_squared_distances(
    const float *x_train,
    const float *x_test,
    float *distances,
    const int n_train,
    const int n_test,
    const int n_features)
{
    const int train_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int test_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (train_idx < n_train && test_idx < n_test)
    {
        float acc = 0.0f;
        const int train_base = train_idx * n_features;
        const int test_base = test_idx * n_features;

        for (int f = 0; f < n_features; ++f)
        {
            const float diff = x_test[test_base + f] - x_train[train_base + f];
            acc += diff * diff;
        }

        distances[test_idx * n_train + train_idx] = acc;
    }
}
"""


BUILTIN_DATASETS = {
    "synthetic",
    "iris",
    "wbc",
    "breast_cancer",
    "wine",
    "digits",
    "mnist",
    "mnist_784",
    "covtype",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU vs PyCUDA brute-force KNN benchmark")

    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--dataset",
        type=str,
        default="iris",
        help=(
            "built-in dataset: iris, wbc, breast_cancer, wine, digits, "
            "mnist_784, covtype, synthetic"
        ),
    )
    source_group.add_argument("--csv", type=str, help="path to a CSV file")

    parser.add_argument("--target-column", type=str, default=None, help="CSV target column name")
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="CSV target column index (default: last column)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter (default: ',')",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="treat CSV as having no header row",
    )

    parser.add_argument("--test-size", type=float, default=0.2, help="test split fraction")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--scale",
        action="store_true",
        default=True,
        help="standardize features using the training split (default: on)",
    )
    parser.add_argument(
        "--no-scale",
        action="store_false",
        dest="scale",
        help="disable standardization",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="optionally cap the number of training samples after the split",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="optionally cap the number of test samples after the split",
    )
    parser.add_argument(
        "--distance-budget-mb",
        type=float,
        default=256.0,
        help="temporary distance-matrix memory budget per batch in MB",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="manually set how many test samples to process per batch",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="skip the slow CPU baseline (useful for very large datasets)",
    )

    parser.add_argument(
        "--block-x", type=int, default=16, help="CUDA block dimension in x (train axis)"
    )
    parser.add_argument(
        "--block-y", type=int, default=16, help="CUDA block dimension in y (test axis)"
    )

    # Synthetic dataset controls
    parser.add_argument("--n-samples", type=int, default=6000, help="synthetic total sample count")
    parser.add_argument("--n-features", type=int, default=32, help="synthetic feature count")
    parser.add_argument("--n-classes", type=int, default=5, help="synthetic class count")

    return parser.parse_args()


# ----------------------------
# Dataset loading
# ----------------------------

def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "This script needs scikit-learn for dataset loading and preprocessing.\n"
            "Install with: pip install scikit-learn"
        ) from exc



def make_synthetic_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=5.0, size=(n_classes, n_features)).astype(np.float32)
    labels = rng.integers(0, n_classes, size=n_samples, dtype=np.int32)
    x = centers[labels] + rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    return x.astype(np.float32), labels.astype(np.int32)



def load_builtin_dataset(name: str, seed: int, n_samples: int, n_features: int, n_classes: int) -> tuple[np.ndarray, np.ndarray, str]:
    require_sklearn()
    from sklearn.datasets import (
        fetch_covtype,
        fetch_openml,
        load_breast_cancer,
        load_digits,
        load_iris,
        load_wine,
    )

    normalized = name.lower()

    if normalized == "synthetic":
        x, y = make_synthetic_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            seed=seed,
        )
        return x, y, "synthetic"

    if normalized == "iris":
        x, y = load_iris(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "iris"

    if normalized in {"wbc", "breast_cancer"}:
        x, y = load_breast_cancer(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "breast_cancer"

    if normalized == "wine":
        x, y = load_wine(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "wine"

    if normalized == "digits":
        x, y = load_digits(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "digits"

    if normalized in {"mnist", "mnist_784"}:
        x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "mnist_784"

    if normalized == "covtype":
        x, y = fetch_covtype(return_X_y=True, as_frame=False)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "covtype"

    raise SystemExit(
        f"Unknown dataset '{name}'. Supported built-ins: {sorted(BUILTIN_DATASETS)}"
    )



def load_csv_dataset(
    csv_path: str,
    target_column: Optional[str],
    target_index: Optional[int],
    delimiter: str,
    has_header: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "CSV mode needs pandas for robust parsing.\n"
            "Install with: pip install pandas"
        ) from exc

    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")

    header = 0 if has_header else None
    df = pd.read_csv(path, delimiter=delimiter, header=header)

    if df.empty:
        raise SystemExit("CSV file is empty.")

    if target_column is not None and target_index is not None:
        raise SystemExit("Choose either --target-column or --target-index, not both.")

    if target_column is not None:
        if target_column not in df.columns:
            raise SystemExit(f"Target column '{target_column}' not found in CSV.")
        y = df[target_column].to_numpy()
        x_df = df.drop(columns=[target_column])
    else:
        if target_index is None:
            target_index = df.shape[1] - 1
        if target_index < 0:
            target_index += df.shape[1]
        if target_index < 0 or target_index >= df.shape[1]:
            raise SystemExit("target column index is out of range")
        y = df.iloc[:, target_index].to_numpy()
        x_df = df.drop(df.columns[target_index], axis=1)

    # Require numeric features.
    numeric = x_df.apply(pd.to_numeric, errors="coerce")
    if numeric.isnull().any().any():
        bad_cols = [str(c) for c in numeric.columns[numeric.isnull().any()].tolist()]
        raise SystemExit(
            "CSV feature columns must be numeric for this KNN benchmark. "
            f"Non-numeric or missing values detected in: {', '.join(bad_cols)}"
        )

    x = numeric.to_numpy(dtype=np.float32)
    return x, y, path.name



def stratified_cap(
    x: np.ndarray,
    y: np.ndarray,
    limit: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(y):
        return x, y
    require_sklearn()
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    try:
        chosen, _ = train_test_split(
            idx,
            train_size=limit,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(idx, size=limit, replace=False)
    chosen = np.sort(chosen)
    return x[chosen], y[chosen]



def prepare_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    require_sklearn()
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if args.csv:
        x, y_raw, dataset_name = load_csv_dataset(
            csv_path=args.csv,
            target_column=args.target_column,
            target_index=args.target_index,
            delimiter=args.delimiter,
            has_header=not args.no_header,
        )
    else:
        x, y_raw, dataset_name = load_builtin_dataset(
            name=args.dataset,
            seed=args.seed,
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_classes=args.n_classes,
        )

    x = np.asarray(x, dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(np.asarray(y_raw)).astype(np.int32)

    if x.ndim != 2:
        raise SystemExit(f"Expected a 2D feature matrix, got shape {x.shape}")
    if len(x) != len(y):
        raise SystemExit("Feature rows and target length do not match.")
    if len(np.unique(y)) < 2:
        raise SystemExit("Need at least two classes for classification.")
    if not (0.0 < args.test_size < 1.0):
        raise SystemExit("test-size must be between 0 and 1")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    x_train, y_train = stratified_cap(x_train, y_train, args.limit_train, args.seed)
    x_test, y_test = stratified_cap(x_test, y_test, args.limit_test, args.seed)

    if args.scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
    else:
        x_train = x_train.astype(np.float32, copy=False)
        x_test = x_test.astype(np.float32, copy=False)

    meta = {
        "dataset_name": dataset_name,
        "original_samples": int(len(x)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(encoder.classes_)),
        "class_names": [str(c) for c in encoder.classes_],
        "scaled": bool(args.scale),
    }
    return x_train, y_train, x_test, y_test, meta


# ----------------------------
# KNN helpers
# ----------------------------

def majority_vote(neighbor_labels: np.ndarray, n_classes: int) -> int:
    counts = np.bincount(neighbor_labels, minlength=n_classes)
    return int(np.argmax(counts))



def predict_from_distances(
    distances: np.ndarray,
    y_train: np.ndarray,
    k: int,
    n_classes: int,
) -> np.ndarray:
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    preds = np.empty(distances.shape[0], dtype=np.int32)
    for i in range(distances.shape[0]):
        preds[i] = majority_vote(y_train[nearest[i]], n_classes)
    return preds



def choose_test_batch_size(
    n_train: int,
    n_test: int,
    budget_mb: float,
    manual_batch: Optional[int],
) -> int:
    if manual_batch is not None:
        return max(1, min(n_test, manual_batch))

    bytes_per_test_row = n_train * np.dtype(np.float32).itemsize
    budget_bytes = max(1, int(budget_mb * 1024 * 1024))
    auto_batch = max(1, budget_bytes // max(1, bytes_per_test_row))
    return max(1, min(n_test, auto_batch))



def iter_test_batches(n_test: int, batch_size: int):
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        yield start, end



def knn_cpu_bruteforce_batched(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    n_classes: int,
    test_batch_size: int,
) -> BenchmarkResult:
    n_test = x_test.shape[0]
    n_train = x_train.shape[0]
    preds = np.empty(n_test, dtype=np.int32)

    start_total = time.perf_counter()

    for start, end in iter_test_batches(n_test, test_batch_size):
        batch = x_test[start:end]
        distances = np.empty((len(batch), n_train), dtype=np.float32)

        for i in range(len(batch)):
            for j in range(n_train):
                diff = batch[i] - x_train[j]
                distances[i, j] = np.dot(diff, diff)

        preds[start:end] = predict_from_distances(distances, y_train, k, n_classes)

    elapsed = time.perf_counter() - start_total

    return BenchmarkResult(
        name="CPU brute-force",
        predictions=preds,
        elapsed_s=elapsed,
        extra={"test_batch_size": int(test_batch_size)},
    )



def build_kernel() -> drv.Function:
    module = SourceModule(CUDA_SRC)
    func = module.get_function("pairwise_squared_distances")
    return func



def warm_up_kernel(func: drv.Function, block: tuple[int, int, int]) -> None:
    x_train = np.zeros((1, 1), dtype=np.float32)
    x_test = np.zeros((1, 1), dtype=np.float32)
    dists = np.zeros((1, 1), dtype=np.float32)

    x_train_gpu = drv.mem_alloc(x_train.nbytes)
    x_test_gpu = drv.mem_alloc(x_test.nbytes)
    dists_gpu = drv.mem_alloc(dists.nbytes)

    drv.memcpy_htod(x_train_gpu, x_train)
    drv.memcpy_htod(x_test_gpu, x_test)

    func(
        x_train_gpu,
        x_test_gpu,
        dists_gpu,
        np.int32(1),
        np.int32(1),
        np.int32(1),
        block=block,
        grid=(1, 1, 1),
    )
    drv.Context.synchronize()



def knn_gpu_distances_batched(
    func: drv.Function,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    n_classes: int,
    test_batch_size: int,
    block_x: int,
    block_y: int,
) -> BenchmarkResult:
    n_train, n_features = x_train.shape
    n_test = x_test.shape[0]
    preds = np.empty(n_test, dtype=np.int32)

    block = (block_x, block_y, 1)
    start_total = time.perf_counter()

    x_train_gpu = drv.mem_alloc(x_train.nbytes)
    drv.memcpy_htod(x_train_gpu, x_train)

    kernel_ms_total = 0.0
    max_batch = min(test_batch_size, n_test)
    x_test_gpu = drv.mem_alloc(max_batch * n_features * np.dtype(np.float32).itemsize)
    distances_gpu = drv.mem_alloc(max_batch * n_train * np.dtype(np.float32).itemsize)

    last_grid = None
    for start, end in iter_test_batches(n_test, test_batch_size):
        x_batch = np.ascontiguousarray(x_test[start:end], dtype=np.float32)
        batch_rows = x_batch.shape[0]
        distances = np.empty((batch_rows, n_train), dtype=np.float32)

        drv.memcpy_htod(x_test_gpu, x_batch)

        grid = (
            (n_train + block_x - 1) // block_x,
            (batch_rows + block_y - 1) // block_y,
            1,
        )
        last_grid = grid

        start_evt = drv.Event()
        end_evt = drv.Event()
        start_evt.record()
        func(
            x_train_gpu,
            x_test_gpu,
            distances_gpu,
            np.int32(n_train),
            np.int32(batch_rows),
            np.int32(n_features),
            block=block,
            grid=grid,
        )
        end_evt.record()
        end_evt.synchronize()
        kernel_ms_total += float(start_evt.time_till(end_evt))

        drv.memcpy_dtoh(distances, distances_gpu)
        preds[start:end] = predict_from_distances(distances, y_train, k, n_classes)

    drv.Context.synchronize()
    elapsed_total = time.perf_counter() - start_total

    return BenchmarkResult(
        name="GPU distances (PyCUDA)",
        predictions=preds,
        elapsed_s=elapsed_total,
        extra={
            "kernel_ms": kernel_ms_total,
            "block": block,
            "grid": last_grid,
            "test_batch_size": int(test_batch_size),
        },
    )


# ----------------------------
# Reporting helpers
# ----------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))



def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"



def main() -> int:
    args = parse_args()

    if args.k < 1:
        raise SystemExit("k must be at least 1")
    if args.block_x * args.block_y > 1024:
        raise SystemExit("block_x * block_y must be <= 1024")
    if args.distance_budget_mb <= 0:
        raise SystemExit("distance-budget-mb must be positive")

    x_train, y_train, x_test, y_test, meta = prepare_dataset(args)

    if args.k > len(y_train):
        raise SystemExit("k cannot be larger than the number of training samples")

    test_batch_size = choose_test_batch_size(
        n_train=len(y_train),
        n_test=len(y_test),
        budget_mb=args.distance_budget_mb,
        manual_batch=args.test_batch_size,
    )

    batch_distance_bytes = test_batch_size * len(y_train) * np.dtype(np.float32).itemsize

    print("=" * 80)
    print("PyCUDA KNN dataset benchmark")
    print("=" * 80)
    print(f"dataset           : {meta['dataset_name']}")
    print(f"original samples  : {meta['original_samples']}")
    print(f"train samples     : {meta['train_samples']}")
    print(f"test samples      : {meta['test_samples']}")
    print(f"features          : {meta['n_features']}")
    print(f"classes           : {meta['n_classes']}")
    print(f"k                 : {args.k}")
    print(f"scaled            : {meta['scaled']}")
    print(f"test batch size   : {test_batch_size}")
    print(f"batch dist matrix : {test_batch_size} x {meta['train_samples']} ({format_bytes(batch_distance_bytes)})")
    print()

    kernel = build_kernel()
    warm_up_kernel(kernel, (args.block_x, args.block_y, 1))

    cpu_result = None
    if not args.skip_cpu:
        cpu_result = knn_cpu_bruteforce_batched(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            k=args.k,
            n_classes=meta['n_classes'],
            test_batch_size=test_batch_size,
        )

    gpu_result = knn_gpu_distances_batched(
        func=kernel,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        k=args.k,
        n_classes=meta['n_classes'],
        test_batch_size=test_batch_size,
        block_x=args.block_x,
        block_y=args.block_y,
    )

    gpu_acc = accuracy(y_test, gpu_result.predictions)

    if cpu_result is not None:
        cpu_acc = accuracy(y_test, cpu_result.predictions)
        agreement = float(np.mean(cpu_result.predictions == gpu_result.predictions))
        speedup = cpu_result.elapsed_s / gpu_result.elapsed_s if gpu_result.elapsed_s > 0 else float("inf")

        print(f"{cpu_result.name:24s}: {cpu_result.elapsed_s:10.6f} s | accuracy = {cpu_acc:.4f}")
        print(
            f"{gpu_result.name:24s}: {gpu_result.elapsed_s:10.6f} s | "
            f"kernel = {gpu_result.extra['kernel_ms']:.3f} ms | accuracy = {gpu_acc:.4f}"
        )
        print(f"prediction agreement    : {agreement:.4f}")
        print(f"end-to-end speedup      : {speedup:.2f}x")
    else:
        print(
            f"{gpu_result.name:24s}: {gpu_result.elapsed_s:10.6f} s | "
            f"kernel = {gpu_result.extra['kernel_ms']:.3f} ms | accuracy = {gpu_acc:.4f}"
        )
        print("CPU brute-force         : skipped")

    print(f"GPU launch config       : block={gpu_result.extra['block']}, grid={gpu_result.extra['grid']}")
    print()
    print("Notes:")
    print("- CPU path is the traditional brute-force nested-loop baseline.")
    print("- GPU path computes pairwise distances on the GPU and keeps KNN voting on the CPU.")
    print("- Kernel compilation time is excluded from the benchmark; it is a one-time setup cost.")
    print("- For huge datasets, use --limit-train / --limit-test to keep the benchmark practical.")
    print("- Datasets like mnist_784 and covtype may download the first time you use them.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
