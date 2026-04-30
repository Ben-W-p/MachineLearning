#!/usr/bin/env python3
"""
pycuda_kmeans_dataset_runner.py

Reusable CPU-vs-GPU k-means clustering benchmark.

What it does
------------
1) CPU baseline: classic k-means with NumPy distance calculations.
2) GPU version: computes point-to-centroid squared distances on the GPU with PyCUDA.
3) Both paths use the same random initialization and CPU centroid update step.

Data sources
------------
- Built-in datasets from scikit-learn:
  iris, wine, digits, mnist_784, covtype, synthetic
- Any CSV file with numeric feature columns
- Optional target column is only used for reporting cluster purity-style agreement,
  not for training

Examples
--------
python pycuda_kmeans_dataset_runner.py --dataset iris --k 3
python pycuda_kmeans_dataset_runner.py --dataset digits --k 10
python pycuda_kmeans_dataset_runner.py --dataset covtype --k 7 --limit-samples 20000
python pycuda_kmeans_dataset_runner.py --csv ./my_points.csv --k 4
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
    from pycuda.compiler import SourceModule
except ImportError as exc:
    raise SystemExit(
        "PyCUDA is not installed or CUDA is not configured correctly.\n"
        "Install with: pip install pycuda\n"
        "Also make sure the CUDA toolkit and driver are installed and nvcc is on PATH."
    ) from exc


@dataclass
class BenchmarkResult:
    name: str
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int
    elapsed_s: float
    extra: dict


CUDA_SRC = r"""
extern "C" __global__ void point_centroid_squared_distances(
    const float *X,
    const float *centroids,
    float *distances,
    const int n_samples,
    const int k,
    const int n_features)
{
    const int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (centroid_idx < k && sample_idx < n_samples)
    {
        float acc = 0.0f;
        const int sample_base = sample_idx * n_features;
        const int centroid_base = centroid_idx * n_features;

        for (int f = 0; f < n_features; ++f)
        {
            float diff = X[sample_base + f] - centroids[centroid_base + f];
            acc += diff * diff;
        }

        distances[sample_idx * k + centroid_idx] = acc;
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
    parser = argparse.ArgumentParser(description="CPU vs PyCUDA k-means benchmark")

    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--dataset",
        type=str,
        default="iris",
        help="built-in dataset: iris, wbc, breast_cancer, wine, digits, mnist_784, covtype, synthetic",
    )
    source_group.add_argument("--csv", type=str, help="path to a CSV file")

    parser.add_argument("--target-column", type=str, default=None, help="optional target column name")
    parser.add_argument("--target-index", type=int, default=None, help="optional target column index")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    parser.add_argument("--no-header", action="store_true", help="treat CSV as having no header row")

    parser.add_argument("--k", type=int, default=None, help="number of clusters (default: infer when possible)")
    parser.add_argument("--max-iter", type=int, default=50, help="max k-means iterations")
    parser.add_argument("--tol", type=float, default=1e-4, help="centroid movement tolerance")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--scale", action="store_true", default=True, help="standardize features")
    parser.add_argument("--no-scale", action="store_false", dest="scale", help="disable standardization")
    parser.add_argument("--limit-samples", type=int, default=None, help="cap total samples")
    parser.add_argument("--distance-budget-mb", type=float, default=256.0, help="distance matrix budget in MB")
    parser.add_argument("--sample-batch-size", type=int, default=None, help="manual sample batch size")
    parser.add_argument("--skip-cpu", action="store_true", help="skip CPU baseline")

    parser.add_argument("--block-x", type=int, default=16, help="CUDA block dimension in x (cluster axis)")
    parser.add_argument("--block-y", type=int, default=16, help="CUDA block dimension in y (sample axis)")

    parser.add_argument("--n-samples", type=int, default=12000, help="synthetic sample count")
    parser.add_argument("--n-features", type=int, default=16, help="synthetic feature count")
    parser.add_argument("--n-clusters", type=int, default=5, help="synthetic cluster count")

    return parser.parse_args()


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script needs scikit-learn for dataset loading and preprocessing.\n"
            "Install with: pip install scikit-learn"
        ) from exc


def make_synthetic_dataset(n_samples: int, n_features: int, n_clusters: int, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=5.0, size=(n_clusters, n_features)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n_samples, dtype=np.int32)
    x = centers[labels] + rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    return x.astype(np.float32), labels.astype(np.int32), n_clusters


def load_builtin_dataset(name: str, seed: int, n_samples: int, n_features: int, n_clusters: int):
    require_sklearn()
    from sklearn.datasets import fetch_covtype, fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine

    normalized = name.lower()

    if normalized == "synthetic":
        x, y, k = make_synthetic_dataset(n_samples, n_features, n_clusters, seed)
        return x, y, "synthetic", k

    if normalized == "iris":
        x, y = load_iris(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), "iris", 3

    if normalized in {"wbc", "breast_cancer"}:
        x, y = load_breast_cancer(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), "breast_cancer", 2

    if normalized == "wine":
        x, y = load_wine(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), "wine", 3

    if normalized == "digits":
        x, y = load_digits(return_X_y=True)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), "digits", 10

    if normalized in {"mnist", "mnist_784"}:
        x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "mnist_784", 10

    if normalized == "covtype":
        x, y = fetch_covtype(return_X_y=True, as_frame=False)
        return np.asarray(x, dtype=np.float32), np.asarray(y), "covtype", 7

    raise SystemExit(f"Unknown dataset '{name}'. Supported built-ins: {sorted(BUILTIN_DATASETS)}")


def load_csv_dataset(csv_path: str, target_column: Optional[str], target_index: Optional[int], delimiter: str, has_header: bool):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("CSV mode needs pandas.\nInstall with: pip install pandas") from exc

    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")

    header = 0 if has_header else None
    df = pd.read_csv(path, delimiter=delimiter, header=header)
    if df.empty:
        raise SystemExit("CSV file is empty.")

    y = None
    x_df = df

    if target_column is not None and target_index is not None:
        raise SystemExit("Choose either --target-column or --target-index, not both.")

    if target_column is not None:
        if target_column not in df.columns:
            raise SystemExit(f"Target column '{target_column}' not found in CSV.")
        y = df[target_column].to_numpy()
        x_df = df.drop(columns=[target_column])
    elif target_index is not None:
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


def cap_samples(x: np.ndarray, y: Optional[np.ndarray], limit: Optional[int], seed: int):
    if limit is None or limit >= len(x):
        return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(len(x)), size=limit, replace=False))
    x2 = x[idx]
    y2 = None if y is None else np.asarray(y)[idx]
    return x2, y2


def prepare_dataset(args: argparse.Namespace):
    require_sklearn()
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if args.csv:
        x, y_raw, dataset_name = load_csv_dataset(
            csv_path=args.csv,
            target_column=args.target_column,
            target_index=args.target_index,
            delimiter=args.delimiter,
            has_header=not args.no_header,
        )
        inferred_k = None
    else:
        x, y_raw, dataset_name, inferred_k = load_builtin_dataset(
            name=args.dataset,
            seed=args.seed,
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_clusters=args.n_clusters,
        )

    x = np.asarray(x, dtype=np.float32)
    y = None
    if y_raw is not None:
        enc = LabelEncoder()
        y = enc.fit_transform(np.asarray(y_raw)).astype(np.int32)
        class_names = [str(c) for c in enc.classes_]
    else:
        class_names = None

    x, y = cap_samples(x, y, args.limit_samples, args.seed)

    if args.scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)

    k = args.k if args.k is not None else inferred_k
    if k is None:
        raise SystemExit("Please provide --k for CSV datasets or datasets without an inferred cluster count.")

    meta = {
        "dataset_name": dataset_name,
        "samples": int(len(x)),
        "n_features": int(x.shape[1]),
        "k": int(k),
        "scaled": bool(args.scale),
        "class_names": class_names,
        "n_classes": None if y is None else int(len(np.unique(y))),
    }
    return x, y, meta


def choose_sample_batch_size(n_samples: int, k: int, budget_mb: float, manual_batch: Optional[int]) -> int:
    if manual_batch is not None:
        return max(1, min(n_samples, manual_batch))
    bytes_per_sample_row = k * np.dtype(np.float32).itemsize
    budget_bytes = max(1, int(budget_mb * 1024 * 1024))
    auto_batch = max(1, budget_bytes // max(1, bytes_per_sample_row))
    return max(1, min(n_samples, auto_batch))


def iter_batches(n_samples: int, batch_size: int):
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield start, end


def init_centroids(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(x)), size=k, replace=False)
    return np.ascontiguousarray(x[idx].copy(), dtype=np.float32)


def assign_labels_cpu(x: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, float]:
    diff = x[:, None, :] - centroids[None, :, :]
    dists = np.sum(diff * diff, axis=2)
    labels = np.argmin(dists, axis=1).astype(np.int32)
    inertia = float(np.sum(dists[np.arange(len(x)), labels]))
    return labels, inertia


def update_centroids_cpu(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    k = len(centroids)
    new_centroids = centroids.copy()
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            new_centroids[c] = np.mean(x[mask], axis=0, dtype=np.float32)
    return np.ascontiguousarray(new_centroids, dtype=np.float32)


def kmeans_cpu(x: np.ndarray, k: int, max_iter: int, tol: float, seed: int) -> BenchmarkResult:
    centroids = init_centroids(x, k, seed)
    start = time.perf_counter()

    for it in range(1, max_iter + 1):
        labels, inertia = assign_labels_cpu(x, centroids)
        new_centroids = update_centroids_cpu(x, labels, centroids)
        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if shift <= tol:
            break

    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        name="CPU k-means",
        labels=labels,
        centroids=centroids,
        inertia=float(inertia),
        iterations=it,
        elapsed_s=elapsed,
        extra={},
    )


def build_kernel():
    module = SourceModule(CUDA_SRC)
    return module.get_function("point_centroid_squared_distances")


def warm_up_kernel(func, block):
    x = np.zeros((1, 1), dtype=np.float32)
    centroids = np.zeros((1, 1), dtype=np.float32)
    d = np.zeros((1, 1), dtype=np.float32)

    x_gpu = drv.mem_alloc(x.nbytes)
    c_gpu = drv.mem_alloc(centroids.nbytes)
    d_gpu = drv.mem_alloc(d.nbytes)

    drv.memcpy_htod(x_gpu, x)
    drv.memcpy_htod(c_gpu, centroids)

    func(
        x_gpu, c_gpu, d_gpu,
        np.int32(1), np.int32(1), np.int32(1),
        block=block, grid=(1, 1, 1),
    )
    drv.Context.synchronize()


def assign_labels_gpu_batched(
    func,
    x: np.ndarray,
    centroids: np.ndarray,
    sample_batch_size: int,
    block_x: int,
    block_y: int,
):
    n_samples, n_features = x.shape
    k = centroids.shape[0]
    labels = np.empty(n_samples, dtype=np.int32)
    inertia = 0.0

    block = (block_x, block_y, 1)
    kernel_ms_total = 0.0

    centroids_gpu = drv.mem_alloc(centroids.nbytes)
    drv.memcpy_htod(centroids_gpu, centroids)

    max_batch = min(sample_batch_size, n_samples)
    x_gpu = drv.mem_alloc(max_batch * n_features * np.dtype(np.float32).itemsize)
    d_gpu = drv.mem_alloc(max_batch * k * np.dtype(np.float32).itemsize)

    last_grid = None
    for start, end in iter_batches(n_samples, sample_batch_size):
        x_batch = np.ascontiguousarray(x[start:end], dtype=np.float32)
        batch_rows = x_batch.shape[0]
        d_host = np.empty((batch_rows, k), dtype=np.float32)

        drv.memcpy_htod(x_gpu, x_batch)

        grid = (
            (k + block_x - 1) // block_x,
            (batch_rows + block_y - 1) // block_y,
            1,
        )
        last_grid = grid

        start_evt = drv.Event()
        end_evt = drv.Event()
        start_evt.record()
        func(
            x_gpu,
            centroids_gpu,
            d_gpu,
            np.int32(batch_rows),
            np.int32(k),
            np.int32(n_features),
            block=block,
            grid=grid,
        )
        end_evt.record()
        end_evt.synchronize()
        kernel_ms_total += float(start_evt.time_till(end_evt))

        drv.memcpy_dtoh(d_host, d_gpu)
        batch_labels = np.argmin(d_host, axis=1).astype(np.int32)
        labels[start:end] = batch_labels
        inertia += float(np.sum(d_host[np.arange(batch_rows), batch_labels]))

    return labels, float(inertia), kernel_ms_total, block, last_grid


def kmeans_gpu(
    func,
    x: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    seed: int,
    sample_batch_size: int,
    block_x: int,
    block_y: int,
) -> BenchmarkResult:
    centroids = init_centroids(x, k, seed)
    kernel_ms_total = 0.0
    last_grid = None
    block = (block_x, block_y, 1)

    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        labels, inertia, kernel_ms, _, last_grid = assign_labels_gpu_batched(
            func=func,
            x=x,
            centroids=centroids,
            sample_batch_size=sample_batch_size,
            block_x=block_x,
            block_y=block_y,
        )
        kernel_ms_total += kernel_ms
        new_centroids = update_centroids_cpu(x, labels, centroids)
        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if shift <= tol:
            break
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="GPU k-means",
        labels=labels,
        centroids=centroids,
        inertia=float(inertia),
        iterations=it,
        elapsed_s=elapsed,
        extra={
            "kernel_ms": kernel_ms_total,
            "block": block,
            "grid": last_grid,
            "sample_batch_size": int(sample_batch_size),
        },
    )


def clustering_accuracy_proxy(y_true: Optional[np.ndarray], labels: np.ndarray) -> Optional[float]:
    if y_true is None:
        return None
    y_true = np.asarray(y_true, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    score = 0
    for cluster in np.unique(labels):
        mask = labels == cluster
        if np.any(mask):
            score += np.bincount(y_true[mask]).max()
    return float(score / len(y_true))


def main() -> int:
    args = parse_args()

    if args.k is not None and args.k < 1:
        raise SystemExit("k must be at least 1")
    if args.max_iter < 1:
        raise SystemExit("max-iter must be at least 1")
    if args.tol < 0:
        raise SystemExit("tol must be non-negative")
    if args.distance_budget_mb <= 0:
        raise SystemExit("distance-budget-mb must be positive")
    if args.block_x * args.block_y > 1024:
        raise SystemExit("block_x * block_y must be <= 1024")

    x, y, meta = prepare_dataset(args)

    if meta["k"] > len(x):
        raise SystemExit("k cannot be larger than the number of samples")

    sample_batch_size = choose_sample_batch_size(
        n_samples=len(x),
        k=meta["k"],
        budget_mb=args.distance_budget_mb,
        manual_batch=args.sample_batch_size,
    )
    batch_distance_bytes = sample_batch_size * meta["k"] * np.dtype(np.float32).itemsize

    print("=" * 80)
    print("PyCUDA k-means dataset benchmark")
    print("=" * 80)
    print(f"dataset           : {meta['dataset_name']}")
    print(f"samples           : {meta['samples']}")
    print(f"features          : {meta['n_features']}")
    print(f"k                 : {meta['k']}")
    print(f"scaled            : {meta['scaled']}")
    print(f"max iterations    : {args.max_iter}")
    print(f"sample batch size : {sample_batch_size}")
    print(f"batch dist matrix : {sample_batch_size} x {meta['k']} ({batch_distance_bytes / (1024*1024):.2f} MB)")
    print()

    kernel = build_kernel()
    warm_up_kernel(kernel, (args.block_x, args.block_y, 1))

    cpu_result = None
    if not args.skip_cpu:
        cpu_result = kmeans_cpu(
            x=x,
            k=meta["k"],
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
        )

    gpu_result = kmeans_gpu(
        func=kernel,
        x=x,
        k=meta["k"],
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        sample_batch_size=sample_batch_size,
        block_x=args.block_x,
        block_y=args.block_y,
    )

    gpu_proxy = clustering_accuracy_proxy(y, gpu_result.labels)

    if cpu_result is not None:
        cpu_proxy = clustering_accuracy_proxy(y, cpu_result.labels)
        agreement = float(np.mean(cpu_result.labels == gpu_result.labels))
        speedup = cpu_result.elapsed_s / gpu_result.elapsed_s if gpu_result.elapsed_s > 0 else float("inf")

        cpu_proxy_text = f" | purity-ish = {cpu_proxy:.4f}" if cpu_proxy is not None else ""
        gpu_proxy_text = f" | purity-ish = {gpu_proxy:.4f}" if gpu_proxy is not None else ""

        print(
            f"{cpu_result.name:20s}: {cpu_result.elapsed_s:10.6f} s | "
            f"iter = {cpu_result.iterations:3d} | inertia = {cpu_result.inertia:.4f}{cpu_proxy_text}"
        )
        print(
            f"{gpu_result.name:20s}: {gpu_result.elapsed_s:10.6f} s | "
            f"kernel = {gpu_result.extra['kernel_ms']:.3f} ms | "
            f"iter = {gpu_result.iterations:3d} | inertia = {gpu_result.inertia:.4f}{gpu_proxy_text}"
        )
        print(f"raw label agreement       : {agreement:.4f}")
        print(f"end-to-end speedup        : {speedup:.2f}x")
    else:
        gpu_proxy_text = f" | purity-ish = {gpu_proxy:.4f}" if gpu_proxy is not None else ""
        print(
            f"{gpu_result.name:20s}: {gpu_result.elapsed_s:10.6f} s | "
            f"kernel = {gpu_result.extra['kernel_ms']:.3f} ms | "
            f"iter = {gpu_result.iterations:3d} | inertia = {gpu_result.inertia:.4f}{gpu_proxy_text}"
        )
        print("CPU k-means           : skipped")

    print(f"GPU launch config     : block={gpu_result.extra['block']}, grid={gpu_result.extra['grid']}")
    print()
    print("Notes:")
    print("- GPU path computes point-to-centroid distances on the GPU.")
    print("- Centroid recomputation is still done on the CPU for simplicity and fair readability.")
    print("- K-means cluster labels can be permuted, so raw label agreement is only a rough indicator.")
    print("- If ground-truth labels exist, 'purity-ish' is reported as an easy clustering proxy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
