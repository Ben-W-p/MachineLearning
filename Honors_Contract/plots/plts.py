#!/usr/bin/env python3
"""
make_benchmark_plots.py

Creates plots from manually entered CPU/GPU benchmark results for:
- KNN
- Logistic Regression
- K-Means

Usage
-----
python3 make_benchmark_plots.py
python3 make_benchmark_plots.py --outdir benchmark_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_time_to_seconds(value):
    if value is None:
        return np.nan
    if isinstance(value, (float, int)):
        return float(value)

    s = str(value).strip().lower()
    if s in {"unknown", "unkown", "na", "n/a", ""}:
        return np.nan

    if s.endswith("ms"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s"):
        return float(s[:-1])

    return float(s)


DATA = {
    "KNN": [
        {"dataset": "Iris", "rows": 150, "features": 4, "cpu": ".0049s", "gpu": ".00026s"},
        {"dataset": "WBC", "rows": 569, "features": 30, "cpu": ".064s", "gpu": ".000743s"},
        {"dataset": "Digits", "rows": 1797, "features": 64, "cpu": ".645s", "gpu": ".004606s"},
        {"dataset": "Covtype", "rows": 581012, "features": 54, "cpu": "UNKNOWN", "gpu": "148s"},
    ],
    "Logistic Regression": [
        {"dataset": "WBC", "rows": 569, "features": 30, "cpu": ".011s", "gpu": ".087s"},
        {"dataset": "synthetic_binary", "rows": 10000, "features": 32, "cpu": ".029s", "gpu": ".13s"},
        {"dataset": "SUSY", "rows": 5000000, "features": 18, "cpu": "24.3s", "gpu": "51.3s"},
    ],
    "K-Means": [
        {"dataset": "Iris", "rows": 150, "features": 4, "cpu": ".00093s", "gpu": "0.001879s"},
        {"dataset": "WBC", "rows": 569, "features": 30, "cpu": ".0021s", "gpu": ".0026s"},
        {"dataset": "Digits", "rows": 1797, "features": 64, "cpu": ".076s", "gpu": ".014s"},
        {"dataset": "Covtype", "rows": 581012, "features": 54, "cpu": "5.02s", "gpu": ".94s"},
        {"dataset": "SUSY", "rows": 5000000, "features": 18, "cpu": "43.5s", "gpu": "21.6s"},
    ],
}


def convert_data(raw_data):
    converted = {}
    for algo, rows in raw_data.items():
        new_rows = []
        for row in rows:
            cpu = parse_time_to_seconds(row["cpu"])
            gpu = parse_time_to_seconds(row["gpu"])
            speedup = cpu / gpu if np.isfinite(cpu) and np.isfinite(gpu) and gpu > 0 else np.nan
            new_rows.append(
                {
                    "dataset": row["dataset"],
                    "rows": int(row["rows"]),
                    "features": int(row["features"]),
                    "cpu": cpu,
                    "gpu": gpu,
                    "speedup": speedup,
                }
            )
        converted[algo] = new_rows
    return converted


def save_algorithm_time_plot(algo_name, rows, outdir):
    valid = [r for r in rows if np.isfinite(r["cpu"]) and np.isfinite(r["gpu"])]
    valid.sort(key=lambda r: r["rows"])
    if not valid:
        return None

    x = np.array([r["rows"] for r in valid], dtype=float)
    cpu = np.array([r["cpu"] for r in valid], dtype=float)
    gpu = np.array([r["gpu"] for r in valid], dtype=float)
    labels = [r["dataset"] for r in valid]

    plt.figure(figsize=(9, 6))
    plt.plot(x, cpu, marker="o", label="CPU time")
    plt.plot(x, gpu, marker="o", label="GPU time")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (rows, log scale)")
    plt.ylabel("Time in seconds (log scale)")
    plt.title(f"{algo_name}: CPU vs GPU Time by Dataset Size")
    plt.legend()
    plt.grid(True, alpha=0.3)

    for xi, yi, label in zip(x, cpu, labels):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5))
    for xi, yi, label in zip(x, gpu, labels):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, -12))

    path = outdir / f"{algo_name.lower().replace(' ', '_').replace('-', '_')}_cpu_vs_gpu_time.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_algorithm_speedup_plot(algo_name, rows, outdir):
    valid = [r for r in rows if np.isfinite(r["speedup"])]
    valid.sort(key=lambda r: r["rows"])
    if not valid:
        return None

    x = np.array([r["rows"] for r in valid], dtype=float)
    y = np.array([r["speedup"] for r in valid], dtype=float)
    labels = [r["dataset"] for r in valid]

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (rows, log scale)")
    plt.ylabel("Speedup = CPU / GPU (log scale)")
    plt.title(f"{algo_name}: GPU Speedup by Dataset Size")
    plt.grid(True, alpha=0.3)

    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5))

    path = outdir / f"{algo_name.lower().replace(' ', '_').replace('-', '_')}_speedup.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_cross_algorithm_speedup_plot(data, outdir):
    plt.figure(figsize=(9, 6))
    added_any = False

    for algo_name, rows in data.items():
        valid = [r for r in rows if np.isfinite(r["speedup"])]
        valid.sort(key=lambda r: r["rows"])
        if not valid:
            continue

        x = np.array([r["rows"] for r in valid], dtype=float)
        y = np.array([r["speedup"] for r in valid], dtype=float)
        plt.plot(x, y, marker="o", label=algo_name)
        added_any = True

    if not added_any:
        plt.close()
        return None

    plt.axhline(1.0, linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (rows, log scale)")
    plt.ylabel("Speedup = CPU / GPU (log scale)")
    plt.title("GPU Speedup vs Dataset Size Across Algorithms")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = outdir / "all_algorithms_speedup_vs_size.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_cpu_gpu_scatter_plot(data, outdir):
    plt.figure(figsize=(9, 6))
    added_any = False

    for algo_name, rows in data.items():
        valid = [r for r in rows if np.isfinite(r["cpu"]) and np.isfinite(r["gpu"])]
        if not valid:
            continue

        cpu = np.array([r["cpu"] for r in valid], dtype=float)
        gpu = np.array([r["gpu"] for r in valid], dtype=float)
        plt.scatter(cpu, gpu, label=algo_name)

        for r in valid:
            plt.annotate(r["dataset"], (r["cpu"], r["gpu"]), textcoords="offset points", xytext=(5, 5))

        added_any = True

    if not added_any:
        plt.close()
        return None

    all_valid = [
        r for rows in data.values() for r in rows
        if np.isfinite(r["cpu"]) and np.isfinite(r["gpu"])
    ]
    vals = np.array([r["cpu"] for r in all_valid] + [r["gpu"] for r in all_valid], dtype=float)
    min_v = np.min(vals) * 0.8
    max_v = np.max(vals) * 1.2
    line = np.array([min_v, max_v], dtype=float)

    plt.plot(line, line, linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("CPU time (seconds, log scale)")
    plt.ylabel("GPU time (seconds, log scale)")
    plt.title("CPU Time vs GPU Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = outdir / "cpu_time_vs_gpu_time_scatter.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_feature_vs_speedup_plot(data, outdir):
    plt.figure(figsize=(9, 6))
    added_any = False

    for algo_name, rows in data.items():
        valid = [r for r in rows if np.isfinite(r["speedup"])]
        if not valid:
            continue

        x = np.array([r["features"] for r in valid], dtype=float)
        y = np.array([r["speedup"] for r in valid], dtype=float)
        plt.scatter(x, y, label=algo_name)

        for r in valid:
            plt.annotate(r["dataset"], (r["features"], r["speedup"]), textcoords="offset points", xytext=(5, 5))

        added_any = True

    if not added_any:
        plt.close()
        return None

    plt.yscale("log")
    plt.xlabel("Number of features")
    plt.ylabel("Speedup = CPU / GPU (log scale)")
    plt.title("GPU Speedup vs Number of Features")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = outdir / "speedup_vs_features.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_summary_csv(data, outdir):
    lines = ["algorithm,dataset,rows,features,cpu_seconds,gpu_seconds,speedup_cpu_div_gpu"]
    for algo_name, rows in data.items():
        for r in rows:
            cpu = "" if not np.isfinite(r["cpu"]) else f"{r['cpu']:.12g}"
            gpu = "" if not np.isfinite(r["gpu"]) else f"{r['gpu']:.12g}"
            speedup = "" if not np.isfinite(r["speedup"]) else f"{r['speedup']:.12g}"
            lines.append(
                f"{algo_name},{r['dataset']},{r['rows']},{r['features']},{cpu},{gpu},{speedup}"
            )

    path = outdir / "benchmark_data_normalized.csv"
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description="Create benchmark plots from CPU/GPU timing data.")
    parser.add_argument("--outdir", default="benchmark_plots", help="output directory for PNG plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = convert_data(DATA)
    created = []

    for algo_name, rows in data.items():
        p1 = save_algorithm_time_plot(algo_name, rows, outdir)
        p2 = save_algorithm_speedup_plot(algo_name, rows, outdir)
        if p1:
            created.append(p1)
        if p2:
            created.append(p2)

    for fn in (
        save_cross_algorithm_speedup_plot,
        save_cpu_gpu_scatter_plot,
        save_feature_vs_speedup_plot,
    ):
        p = fn(data, outdir)
        if p:
            created.append(p)

    created.append(save_summary_csv(data, outdir))

    print("Created files:")
    for p in created:
        print(p)


if __name__ == "__main__":
    main()