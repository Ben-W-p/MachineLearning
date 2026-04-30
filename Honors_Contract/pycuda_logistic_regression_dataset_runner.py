#!/usr/bin/env python3
"""
pycuda_logistic_regression_dataset_runner.py

Reusable CPU-vs-GPU binary logistic regression benchmark.

What it does
------------
1) CPU baseline: full-batch gradient descent with NumPy.
2) GPU version: computes logits, sigmoid probabilities, gradients, and loss on the GPU
   with PyCUDA each iteration.
3) Both paths use the same data split and preprocessing so training and evaluation are comparable.

Data sources
------------
- Built-in datasets from scikit-learn:
  wbc (breast cancer), breast_cancer, synthetic_binary
- Any CSV file with numeric feature columns and a binary target column

Examples
--------
# Wisconsin breast cancer dataset
python pycuda_logistic_regression_dataset_runner.py --dataset wbc

# Synthetic binary dataset
python pycuda_logistic_regression_dataset_runner.py --dataset synthetic_binary --n-samples 20000 --n-features 64

# Your own CSV, using the last column as the target
python pycuda_logistic_regression_dataset_runner.py --csv ./my_binary_data.csv

# Your own CSV, specifying the target column name
python pycuda_logistic_regression_dataset_runner.py --csv ./wbc.csv --target-column diagnosis

Notes
-----
- This benchmark is for binary classification only.
- CSV features must be numeric. String binary labels are okay; they are label-encoded.
- CPU and GPU both use full-batch gradient descent for a clean apples-to-apples comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
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
    weights: np.ndarray
    bias: float
    train_losses: np.ndarray
    elapsed_s: float
    extra: dict


CUDA_SRC = r"""
extern "C" {

__global__ void compute_logits(
    const float *X,
    const float *w,
    float b,
    float *logits,
    int n_samples,
    int n_features)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float acc = b;
        int base = i * n_features;
        for (int j = 0; j < n_features; ++j) {
            acc += X[base + j] * w[j];
        }
        logits[i] = acc;
    }
}

__global__ void sigmoid_and_diff(
    const float *logits,
    const float *y,
    float *pred,
    float *diff,
    int n_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float z = logits[i];
        float p = 1.0f / (1.0f + expf(-z));
        pred[i] = p;
        diff[i] = p - y[i];
    }
}

__global__ void gradient_w(
    const float *X,
    const float *diff,
    float *grad_w,
    int n_samples,
    int n_features)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_features) {
        float acc = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            acc += X[i * n_features + j] * diff[i];
        }
        grad_w[j] = acc / (float)n_samples;
    }
}

__global__ void reduce_sum(
    const float *x,
    float *out,
    int n)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < n) val = x[i];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void compute_log_loss_terms(
    const float *pred,
    const float *y,
    float *terms,
    int n_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float p = pred[i];
        if (p < 1e-7f) p = 1e-7f;
        if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
        float yi = y[i];
        terms[i] = -(yi * logf(p) + (1.0f - yi) * logf(1.0f - p));
    }
}

}
"""


BUILTIN_DATASETS = {
    "wbc",
    "breast_cancer",
    "synthetic_binary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU vs PyCUDA logistic regression benchmark")

    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--dataset",
        type=str,
        default="wbc",
        help="built-in dataset: wbc, breast_cancer, synthetic_binary",
    )
    source_group.add_argument("--csv", type=str, help="path to a CSV file")

    parser.add_argument("--target-column", type=str, default=None, help="CSV target column name")
    parser.add_argument("--target-index", type=int, default=None, help="CSV target column index")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    parser.add_argument("--no-header", action="store_true", help="treat CSV as having no header row")

    parser.add_argument("--test-size", type=float, default=0.2, help="test split fraction")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--scale", action="store_true", default=True, help="standardize features")
    parser.add_argument("--no-scale", action="store_false", dest="scale", help="disable standardization")
    parser.add_argument("--limit-train", type=int, default=None, help="cap training samples")
    parser.add_argument("--limit-test", type=int, default=None, help="cap test samples")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="gradient descent step size")
    parser.add_argument("--epochs", type=int, default=300, help="number of full-batch epochs")
    parser.add_argument("--threshold", type=float, default=0.5, help="classification threshold")
    parser.add_argument("--skip-cpu", action="store_true", help="skip CPU baseline")

    parser.add_argument("--block-size", type=int, default=256, help="CUDA block size")

    parser.add_argument("--n-samples", type=int, default=10000, help="synthetic sample count")
    parser.add_argument("--n-features", type=int, default=32, help="synthetic feature count")

    return parser.parse_args()


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script needs scikit-learn for dataset loading and preprocessing.\n"
            "Install with: pip install scikit-learn"
        ) from exc


def make_synthetic_binary_dataset(
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=n_features).astype(np.float32)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    logits = x @ true_w + 0.25 * rng.normal(size=n_samples).astype(np.float32)
    y = (logits > 0.0).astype(np.int32)
    return x, y


def load_builtin_dataset(name: str, seed: int, n_samples: int, n_features: int) -> tuple[np.ndarray, np.ndarray, str]:
    require_sklearn()
    from sklearn.datasets import load_breast_cancer

    normalized = name.lower()
    if normalized == "synthetic_binary":
        x, y = make_synthetic_binary_dataset(n_samples=n_samples, n_features=n_features, seed=seed)
        return x.astype(np.float32), y.astype(np.int32), "synthetic_binary"

    if normalized in {"wbc", "breast_cancer"}:
        x, y = load_breast_cancer(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), "breast_cancer"

    raise SystemExit(f"Unknown dataset '{name}'. Supported built-ins: {sorted(BUILTIN_DATASETS)}")


def load_csv_dataset(
    csv_path: str,
    target_column: Optional[str],
    target_index: Optional[int],
    delimiter: str,
    has_header: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import pandas as pd
    except ImportError as exc:
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

    numeric = x_df.apply(pd.to_numeric, errors="coerce")
    if numeric.isnull().any().any():
        bad_cols = [str(c) for c in numeric.columns[numeric.isnull().any()].tolist()]
        raise SystemExit(
            "CSV feature columns must be numeric. "
            f"Non-numeric or missing values detected in: {', '.join(bad_cols)}"
        )

    x = numeric.to_numpy(dtype=np.float32)
    return x, y, path.name


def stratified_cap(x: np.ndarray, y: np.ndarray, limit: Optional[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(y):
        return x, y
    require_sklearn()
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    try:
        chosen, _ = train_test_split(idx, train_size=limit, random_state=seed, stratify=y)
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
        )

    x = np.asarray(x, dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(np.asarray(y_raw)).astype(np.int32)

    if len(np.unique(y)) != 2:
        raise SystemExit(
            f"Logistic regression script expects exactly 2 classes, got {len(np.unique(y))}."
        )

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
        "class_names": [str(c) for c in encoder.classes_],
        "scaled": bool(args.scale),
    }
    return x_train, y_train.astype(np.float32), x_test, y_test.astype(np.int32), meta


def sigmoid_cpu(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def predict_binary(x: np.ndarray, w: np.ndarray, b: float, threshold: float) -> np.ndarray:
    probs = sigmoid_cpu(x @ w + b)
    return (probs >= threshold).astype(np.int32)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def build_kernels():
    module = SourceModule(CUDA_SRC)
    return {
        "compute_logits": module.get_function("compute_logits"),
        "sigmoid_and_diff": module.get_function("sigmoid_and_diff"),
        "gradient_w": module.get_function("gradient_w"),
        "reduce_sum": module.get_function("reduce_sum"),
        "compute_log_loss_terms": module.get_function("compute_log_loss_terms"),
    }


def gpu_reduce_sum(arr_gpu: gpuarray.GPUArray, reduce_kernel, block_size: int) -> float:
    current = arr_gpu
    while current.size > 1:
        blocks = int((current.size + block_size - 1) // block_size)
        out = gpuarray.empty((blocks,), np.float32)
        shared = block_size * np.dtype(np.float32).itemsize
        reduce_kernel(
            current,
            out,
            np.int32(current.size),
            block=(block_size, 1, 1),
            grid=(blocks, 1, 1),
            shared=shared,
        )
        current = out
    return float(current.get()[0])


def logistic_regression_cpu(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
) -> BenchmarkResult:
    n_samples, n_features = x_train.shape
    w = np.zeros(n_features, dtype=np.float32)
    b = np.float32(0.0)
    losses = np.empty(epochs, dtype=np.float32)

    start = time.perf_counter()
    for epoch in range(epochs):
        logits = x_train @ w + b
        probs = sigmoid_cpu(logits)
        diff = probs - y_train
        grad_w = (x_train.T @ diff) / n_samples
        grad_b = np.mean(diff)

        w -= learning_rate * grad_w.astype(np.float32)
        b = np.float32(b - learning_rate * grad_b)
        losses[epoch] = binary_log_loss(y_train, probs)

    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        name="CPU logistic regression",
        weights=w,
        bias=float(b),
        train_losses=losses,
        elapsed_s=elapsed,
        extra={},
    )


def logistic_regression_gpu(
    kernels,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    block_size: int,
) -> BenchmarkResult:
    n_samples, n_features = x_train.shape

    x_gpu = gpuarray.to_gpu(np.ascontiguousarray(x_train, dtype=np.float32).ravel())
    y_gpu = gpuarray.to_gpu(np.ascontiguousarray(y_train, dtype=np.float32))
    w_gpu = gpuarray.zeros((n_features,), np.float32)
    logits_gpu = gpuarray.empty((n_samples,), np.float32)
    pred_gpu = gpuarray.empty((n_samples,), np.float32)
    diff_gpu = gpuarray.empty((n_samples,), np.float32)
    grad_w_gpu = gpuarray.empty((n_features,), np.float32)
    loss_terms_gpu = gpuarray.empty((n_samples,), np.float32)

    losses = np.empty(epochs, dtype=np.float32)
    b = np.float32(0.0)

    sample_grid = ((n_samples + block_size - 1) // block_size, 1, 1)
    feature_grid = ((n_features + block_size - 1) // block_size, 1, 1)

    kernel_ms_total = 0.0
    start_total = time.perf_counter()

    for epoch in range(epochs):
        start_evt = drv.Event()
        end_evt = drv.Event()
        start_evt.record()

        kernels["compute_logits"](
            x_gpu,
            w_gpu,
            np.float32(b),
            logits_gpu,
            np.int32(n_samples),
            np.int32(n_features),
            block=(block_size, 1, 1),
            grid=sample_grid,
        )

        kernels["sigmoid_and_diff"](
            logits_gpu,
            y_gpu,
            pred_gpu,
            diff_gpu,
            np.int32(n_samples),
            block=(block_size, 1, 1),
            grid=sample_grid,
        )

        kernels["gradient_w"](
            x_gpu,
            diff_gpu,
            grad_w_gpu,
            np.int32(n_samples),
            np.int32(n_features),
            block=(block_size, 1, 1),
            grid=feature_grid,
        )

        kernels["compute_log_loss_terms"](
            pred_gpu,
            y_gpu,
            loss_terms_gpu,
            np.int32(n_samples),
            block=(block_size, 1, 1),
            grid=sample_grid,
        )

        end_evt.record()
        end_evt.synchronize()
        kernel_ms_total += float(start_evt.time_till(end_evt))

        grad_b = gpu_reduce_sum(diff_gpu, kernels["reduce_sum"], block_size) / n_samples
        grad_w = grad_w_gpu.get()

        w_host = w_gpu.get()
        w_host -= np.float32(learning_rate) * grad_w.astype(np.float32)
        w_gpu.set(w_host)

        b = np.float32(b - np.float32(learning_rate) * grad_b)
        losses[epoch] = gpu_reduce_sum(loss_terms_gpu, kernels["reduce_sum"], block_size) / n_samples

    drv.Context.synchronize()
    elapsed = time.perf_counter() - start_total

    return BenchmarkResult(
        name="GPU logistic regression",
        weights=w_gpu.get(),
        bias=float(b),
        train_losses=losses,
        elapsed_s=elapsed,
        extra={"kernel_ms": kernel_ms_total, "block_size": int(block_size)},
    )


def main() -> int:
    args = parse_args()

    if args.epochs < 1:
        raise SystemExit("epochs must be at least 1")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive")
    if not (0.0 < args.threshold < 1.0):
        raise SystemExit("threshold must be between 0 and 1")
    if not (1 <= args.block_size <= 1024):
        raise SystemExit("block-size must be between 1 and 1024")

    x_train, y_train, x_test, y_test, meta = prepare_dataset(args)

    print("=" * 80)
    print("PyCUDA logistic regression dataset benchmark")
    print("=" * 80)
    print(f"dataset           : {meta['dataset_name']}")
    print(f"original samples  : {meta['original_samples']}")
    print(f"train samples     : {meta['train_samples']}")
    print(f"test samples      : {meta['test_samples']}")
    print(f"features          : {meta['n_features']}")
    print(f"classes           : 2")
    print(f"scaled            : {meta['scaled']}")
    print(f"epochs            : {args.epochs}")
    print(f"learning rate     : {args.learning_rate}")
    print()

    kernels = build_kernels()

    cpu_result = None
    if not args.skip_cpu:
        cpu_result = logistic_regression_cpu(
            x_train=x_train,
            y_train=y_train,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

    gpu_result = logistic_regression_gpu(
        kernels=kernels,
        x_train=x_train,
        y_train=y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
    )

    gpu_pred = predict_binary(x_test, gpu_result.weights, gpu_result.bias, args.threshold)
    gpu_acc = accuracy(y_test, gpu_pred)

    if cpu_result is not None:
        cpu_pred = predict_binary(x_test, cpu_result.weights, cpu_result.bias, args.threshold)
        cpu_acc = accuracy(y_test, cpu_pred)
        agreement = float(np.mean(cpu_pred == gpu_pred))
        speedup = cpu_result.elapsed_s / gpu_result.elapsed_s if gpu_result.elapsed_s > 0 else float("inf")

        print(
            f"{cpu_result.name:28s}: {cpu_result.elapsed_s:10.6f} s | "
            f"final loss = {cpu_result.train_losses[-1]:.6f} | accuracy = {cpu_acc:.4f}"
        )
        print(
            f"{gpu_result.name:28s}: {gpu_result.elapsed_s:10.6f} s | "
            f"kernel = {gpu_result.extra['kernel_ms']:.3f} ms | "
            f"final loss = {gpu_result.train_losses[-1]:.6f} | accuracy = {gpu_acc:.4f}"
        )
        print(f"prediction agreement        : {agreement:.4f}")
        print(f"end-to-end speedup          : {speedup:.2f}x")
    else:
        print(
            f"{gpu_result.name:28s}: {gpu_result.elapsed_s:10.6f} s | "
            f"kernel = {gpu_result.extra['kernel_ms']:.3f} ms | "
            f"final loss = {gpu_result.train_losses[-1]:.6f} | accuracy = {gpu_acc:.4f}"
        )
        print("CPU logistic regression     : skipped")

    print(f"GPU block size              : {gpu_result.extra['block_size']}")
    print()
    print("Notes:")
    print("- This script is binary logistic regression only.")
    print("- CPU and GPU both use full-batch gradient descent.")
    print("- GPU timing includes all training iterations, not just one kernel launch.")
    print("- For large synthetic problems, use --skip-cpu if the CPU baseline gets too slow.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
