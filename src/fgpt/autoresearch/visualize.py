"""
Visualize experiment progress: val loss over time, annotated with git tags.

Usage:
    python -m fgpt.autoresearch.visualize

Reads experiments/results.jsonl and writes experiments/plots/progress.png.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_FILE = REPO_ROOT / "experiments" / "results.jsonl"
PLOTS_DIR = REPO_ROOT / "experiments" / "plots"


def load_results(path=RESULTS_FILE):
    if not path.exists():
        print(f"No results found at {path}. Run some experiments first.")
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def plot(results, out_path=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed — skipping plot. Run: pip install matplotlib")
        return

    if not results:
        print("No results to plot.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = PLOTS_DIR / "progress.png"

    ys = [r["val_loss"] for r in results]
    tags = [r["tag"] for r in results]
    descriptions = [r.get("description", "") for r in results]

    # Compute x as hours elapsed since first experiment started
    t0 = datetime.fromisoformat(results[0]["timestamp"]) - timedelta(
        seconds=results[0].get("duration_s", 0)
    )

    def end_hours(r):
        ts = datetime.fromisoformat(r["timestamp"])
        return (ts - t0).total_seconds() / 3600

    xs = [end_hours(r) for r in results]

    fig, ax = plt.subplots(figsize=(max(10, len(results) * 1.4), 6))

    # Phase background shading
    phase_boundary = 7.41  # hours
    x_min = min(xs) - 0.5
    x_max = max(xs) + 1.5
    ax.axvspan(x_min, phase_boundary, alpha=0.06, color="#4575b4", zorder=0)
    ax.axvspan(phase_boundary, x_max, alpha=0.06, color="#d73027", zorder=0)
    ax.axvline(phase_boundary, color="#888888", linewidth=1.2, linestyle="--", zorder=2)
    # Phase labels use blended transform: x in data coords, y in axes fraction
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        phase_boundary - 0.1,
        0.02,
        "Phase 1",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4575b4",
        fontstyle="italic",
        transform=trans,
    )
    ax.text(
        phase_boundary + 0.1,
        0.02,
        "Phase 2",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#d73027",
        fontstyle="italic",
        transform=trans,
    )

    ax.plot(xs, ys, marker="o", linewidth=2, markersize=8, color="#2c7bb6", zorder=3)

    # Annotate each point with tag + short description
    best_loss = min(ys)
    for x, y, tag, desc in zip(xs, ys, tags, descriptions):
        is_best = y == best_loss
        color = "#d7191c" if is_best else "#444444"
        label = f"{tag}\n{desc}" if desc else tag
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            fontweight="bold" if is_best else "normal",
            rotation=25,
        )

    # Highlight best
    best_x = xs[ys.index(best_loss)]
    ax.scatter(
        [best_x],
        [best_loss],
        s=120,
        color="#d7191c",
        zorder=4,
        label=f"Best: {best_loss:.4f}",
    )

    ax.set_xlabel("Time elapsed (hours)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(
        "FGPT Autoresearch — Val Loss Progress", fontsize=14, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fh"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def main():
    results = load_results()
    if results:
        print(f"Found {len(results)} experiments:")
        for i, r in enumerate(results):
            print(
                f"  [{i}] {r['tag']:35s}  val_loss={r['val_loss']:.4f}  ({r['description']})"
            )
        print()
    plot(results)


if __name__ == "__main__":
    main()
