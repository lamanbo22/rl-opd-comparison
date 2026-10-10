"""Plot seed accuracies and paired differences from the released results."""
import argparse
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def rows(directory, name):
    with (directory / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/results.png")
    args = parser.parse_args()
    methods = rows(args.results_dir, "method_results.csv")
    contrasts = rows(args.results_dir, "paired_contrasts.csv")
    evaluations = rows(args.results_dir, "evaluation_results.csv")
    labels = {"rl": "RL", "opd": "OPD", "joint": "Joint RL + OPD",
              "opd_then_rl": "OPD → RL", "rl_then_opd": "RL → OPD",
              "initial_student": "Initial student"}
    by_method = {row["method"]: row for row in methods}
    order = list(labels)[:-1]
    initial = next(row for row in evaluations
                   if row["split"] == "test" and row["name"] == "verl_study_initial_test_01")
    baseline = 100 * int(initial["correct"]) / int(initial["questions"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": .49})
    blue, teal, ink = "#4779A7", "#087F8C", "#233047"
    for i, method in enumerate(order):
        row = by_method[method]
        color = teal if method == "rl_then_opd" else blue
        for offset, seed in zip([-.13, 0, .13], [42, 43, 44]):
            ax.scatter(float(row[f"seed{seed}_accuracy_percent"]), i + offset,
                       s=48, color=color, alpha=.85, zorder=3)
        ax.scatter(float(row["mean_accuracy_percent"]), i, marker="D", s=47,
                   facecolor=ink, edgecolor="white", linewidth=.6, zorder=4)
    ax.axvline(baseline, color="#7C858F", linestyle="--", linewidth=1.3)
    ax.text(baseline + .06, -.53, f"Initial student: {baseline:.2f}%", fontsize=9, color="#58636F")
    ax.set(yticks=range(5), yticklabels=[labels[x] for x in order],
           xlabel="Official test accuracy (%)", xlim=(77.9, 81.45), ylim=(4.55, -.9))
    ax.set_title("Accuracy across three training seeds", loc="left", pad=20, weight="bold")
    ax.grid(axis="x", alpha=.15)
    ax.legend(handles=[Line2D([], [], marker="o", color=blue, linestyle="none", label="Training seed"),
                       Line2D([], [], marker="D", color=ink, linestyle="none", label="Mean")],
              loc="lower left", bbox_to_anchor=(0, -.32), ncol=2, frameon=False, fontsize=10)

    comparators = ["rl", "opd", "joint", "opd_then_rl", "initial_student"]
    for i, comparator in enumerate(comparators):
        if comparator == "initial_student":
            row = next(r for r in contrasts if r["left"] == "rl_then_opd" and r["right"] == comparator)
            value, low, high = [float(row[k]) for k in ("difference_pp", "ci_low_pp", "ci_high_pp")]
        else:
            row = next(r for r in contrasts if r["left"] == comparator and r["right"] == "rl_then_opd")
            value = -float(row["difference_pp"])
            low, high = -float(row["ci_high_pp"]), -float(row["ci_low_pp"])
        bx.errorbar(value, i, xerr=[[value - low], [high - value]], fmt="o",
                    color=teal, capsize=4, elinewidth=2, markersize=6)
        bx.text(value, i - .19, f"{value:+.2f} pp", ha="center", fontsize=9, color=ink)
    bx.axvline(0, color="#7C858F", linestyle="--", linewidth=1.3)
    bx.set(yticks=range(5), yticklabels=["vs " + labels[x] for x in comparators],
           xlabel="Accuracy difference (percentage points)", xlim=(-1.05, 4.35), ylim=(4.55, -.9))
    bx.set_title("RL → OPD: paired comparisons", loc="left", pad=20, weight="bold")
    bx.grid(axis="x", alpha=.15)
    fig.subplots_adjust(left=.115, right=.985, top=.83, bottom=.25)
    fig.text(.115, .035, "1,319 test questions · 3 training seeds · Intervals: descriptive, unadjusted 95% crossed bootstrap",
             fontsize=10, color="#58636F")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
