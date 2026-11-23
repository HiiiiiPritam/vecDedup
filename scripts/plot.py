"""Plot VecDedup benchmark results for the report / presentation.

Usage:  python scripts/plot.py [results/results.json]
Writes PNGs next to the JSON file. Needs matplotlib.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (fixed order; never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False,
})


def method_color(methods):
    return {m: SERIES[i] for i, m in enumerate(methods)}  # colour follows the entity


def plot_vectors(res, outdir):
    datasets = []
    for s in res["scenarios"]:
        if s["dataset"] not in datasets:
            datasets.append(s["dataset"])
    methods = [m["method"] for m in res["scenarios"][0]["methods"] if m["method"] != "full-copy"]
    colors = method_color(methods)
    for ds in datasets:
        rows = [s for s in res["scenarios"] if s["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(11, 4.8))
        n = len(methods)
        width = 0.8 / n
        for j, m in enumerate(methods):
            xs, ys = [], []
            for i, s in enumerate(rows):
                mr = next(x for x in s["methods"] if x["method"] == m)
                xs.append(i - 0.4 + width * (j + 0.5))
                ys.append(max(mr["incremental_ratio"], 1e-3))
            ax.bar(xs, ys, width=width * 0.85, color=colors[m], label=m, edgecolor=SURF, linewidth=1)
        ax.set_yscale("log")
        ax.axhline(1, color=BASE, linewidth=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([s["scenario"].replace("-", "\n", 1) for s in rows])
        ax.set_ylabel("incremental dedup ratio\n(log scale, higher = better)")
        ax.set_title(f"Cost of each new snapshot: {ds}", loc="left", fontweight="bold")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"vectors_{ds}.png"), dpi=160)
        plt.close(fig)


def plot_graph(res, outdir):
    if not res.get("graph"):
        return
    methods = ["zstd", "fixed128K+zstd", "fastcdc8K+zstd", "vecdedup(vectors+TGIS)"]
    colors = {"zstd": SERIES[0], "fixed128K+zstd": SERIES[1], "fastcdc8K+zstd": SERIES[2], "vecdedup(vectors+TGIS)": SERIES[5]}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs = [i - 0.4 + width * (j + 0.5) for i in range(len(res["graph"]))]
        ys = [g["increment_bytes"][m] / 2**20 for g in res["graph"]]
        ax.bar(xs, ys, width=width * 0.85, color=colors[m], label=m, edgecolor=SURF)
    for i, g in enumerate(res["graph"]):
        v = g["increment_bytes"]["vecdedup(vectors+TGIS)"] / 2**20
        ax.annotate(f"{v:.2f} MiB", (i - 0.4 + width * 3.5, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(res["graph"])))
    ax.set_xticklabels([f"{g['churn']*100:.0f}% churn" for g in res["graph"]])
    ax.set_ylabel("bytes added by the new snapshot (MiB)")
    ax.set_title(f"HNSW index snapshot cost: {res['graph'][0]['dataset']}", loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "graph_snapshots.png"), dpi=160)
    plt.close(fig)


def plot_lossy(res, outdir):
    if not res.get("lossy"):
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    dss = []
    for p in res["lossy"]:
        if p["dataset"] not in dss:
            dss.append(p["dataset"])
    for i, ds in enumerate(dss):
        pts = sorted([p for p in res["lossy"] if p["dataset"] == ds], key=lambda p: p["compression"])
        xs = [p["compression"] for p in pts]
        ys = [p["recall10"] for p in pts]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURF, markeredgewidth=2, label=ds)
        last = pts[-1]
        ax.annotate(f"M={last['pq_m']}", (last["compression"], last["recall10"]), textcoords="offset points",
                    xytext=(6, -4), fontsize=8, color=INK2)
    ax.set_xlabel("compression vs. raw float32 (×)")
    ax.set_ylabel("Recall@10 (exact search on reconstructed)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Lossy mode: compression ↔ recall trade-off", loc="left", fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lossy_tradeoff.png"), dpi=160)
    plt.close(fig)


def plot_latency(res, outdir):
    if not res.get("latency"):
        return
    labels = ["full-scan commit", "incremental commit", "branch", "checkout"]
    keys = ["full_commit_ms", "incremental_commit_ms", "branch_ms", "checkout_ms"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(res["latency"])
    for j, l in enumerate(res["latency"]):
        xs = [i - 0.4 + width * (j + 0.5) for i in range(len(keys))]
        ax.bar(xs, [max(l[k], 0.01) for k in keys], width=width * 0.85, color=SERIES[j],
               label=f"{l['n']:,} vectors ({l['changed']} changed)", edgecolor=SURF)
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("milliseconds (log scale)")
    ax.set_title("Operation latency", loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latency.png"), dpi=160)
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    outdir = os.path.dirname(path) or "."
    if res.get("scenarios"):
        plot_vectors(res, outdir)
    plot_graph(res, outdir)
    plot_lossy(res, outdir)
    plot_latency(res, outdir)
    print("plots written to", outdir)


if __name__ == "__main__":
    main()
