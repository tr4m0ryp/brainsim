"""Render the README figures from data.json.

Docs only: brainsim never imports this file, and matplotlib is an optional
dependency (`pip install 'brainsim[figures]'`). Every plotted value is read
from data.json, which names the source file and field of each one.

    python3 docs/figures/plot.py              # PNG (light and dark, 2x) and SVG next to this file
    python3 docs/figures/plot.py --pdf DIR    # also a light PDF per figure into DIR
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "quiet": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "quiet": "#52514e"},
}


def style(t):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": t["ink2"], "lines.linewidth": 2, "lines.markersize": 7,
    })


def capacity(d, t):
    rows = d["rows"][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    color = {"measured wiring": t["s2"], "control": t["s1"]}
    seen = set()
    for y, r in enumerate(rows):
        lab = None if r["group"] in seen else r["group"]
        seen.add(r["group"])
        lo, hi = r["ci95"]
        ax.errorbar(r["mean"], y, xerr=[[r["mean"] - lo], [hi - r["mean"]]], fmt="o", color=color[r["group"]],
                    ecolor=color[r["group"]], elinewidth=1.5, capsize=0, label=lab)
        ax.text(hi + 0.6, y, f"{r['mean']:.2f}", va="center", color=t["ink2"], fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.5, 38)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("capacity C, patterns at mean recall overlap 0.9 (95% interval)")
    ax.set_title(d["title"])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center right", fontsize=8.5)
    fig.tight_layout()
    return fig


def ratio(d, t):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    lo, hi = d["kill_band"]
    ax.axvspan(lo, hi, color=t["quiet"], alpha=0.45, lw=0)
    ax.text((lo + hi) / 2, 5.55, "kill band\n[0.95, 1.05]", ha="center", va="top", color=t["ink2"], fontsize=8)
    ax.axvline(d["claim_bar"], color=t["muted"], lw=1, ls=(0, (3, 2)))
    ax.text(d["claim_bar"] + 0.02, 5.55, "claim bar\n1.10", ha="left", va="top", color=t["ink2"], fontsize=8)
    for i, (s, v) in enumerate(zip(d["seeds"], d["per_seed"])):
        ax.plot(v, i + 1, "o", color=t["s2"])
        ax.text(v + 0.03, i + 1, f"{v:.4f}", va="center", color=t["ink2"], fontsize=8.5)
    m, (clo, chi) = d["mean"], d["ci95"]
    ax.errorbar(m, 0, xerr=[[m - clo], [chi - m]], fmt="D", color=t["s1"], ecolor=t["s1"], elinewidth=1.5,
                capsize=0)
    ax.text(chi + 0.03, 0, f"mean {m:.4f} [{clo:.4f}, {chi:.4f}]", va="center", color=t["ink2"], fontsize=8.5)
    ax.set_yticks(range(len(d["seeds"]) + 1), ["mean"] + [f"seed {s}" for s in d["seeds"]])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.05, 1.45)
    ax.set_ylim(-0.6, len(d["seeds"]) + 0.8)
    ax.set_xlabel("C(measured) / mean C(5 degree-preserving shuffles)")
    ax.set_title(d["title"])
    fig.tight_layout()
    return fig


def gate(d, t):
    conds = d["conditions"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    w = 0.36
    series = [("Shiu et al. model (Brian2)", d["shiu"]["values"], t["s1"], -w / 2),
              ("brainsim torch engine", d["brainsim"]["values"], t["s2"], w / 2)]
    for lab, vals, c, off in series:
        xs = [i + off for i in range(len(conds))]
        ax.bar(xs, vals, width=w - 0.04, color=c, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 3, f"{v:.1f}", ha="center", va="bottom", color=t["ink2"], fontsize=8)
    ax.set_xticks(range(len(conds)), conds)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 195)
    ax.set_ylabel("MN9 spikes per trial")
    ax.set_title(d["title"])
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


def loop(d, t):
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    stages = d["stages"]
    x = {s["stage"]: 1.0 + i * 2.2 for i, s in enumerate(stages)}
    y, bw, bh = 2.0, 1.55, 1.0
    rules = {"btsp": "BTSP", "mb": "MB rule", "cascade": "cascade"}
    for s in stages:
        ax.add_patch(FancyBboxPatch((x[s["stage"]] - bw / 2, y - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.02,rounding_size=0.12", fc=t["surface"],
                                    ec=t["s1"], lw=1.5))
        ax.text(x[s["stage"]], y + 0.15, s["label"].replace(" (", "\n("), ha="center", va="center",
                fontsize=7.3, color=t["ink"])
        ax.text(x[s["stage"]], y - 0.33, f"{s['n']:,}", ha="center", va="center", fontsize=8,
                color=t["ink2"], fontweight="bold")

    def arrow(p0, p1, rad, color):
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=10, color=color, lw=1.2))

    for e in d["edges"]:
        c = t["s2"] if e["plastic"] else t["muted"]
        lab = f"k {e['k']:,}" + (f"\n{rules[e['plastic']]}" if e["plastic"] else "")
        x0, x1 = x[e["pre"]], x[e["post"]]
        if e["pre"] == e["post"]:
            arrow((x0 - 0.35, y + bh / 2), (x0 + 0.35, y + bh / 2), -1.6, c)
            ax.text(x0, y + bh / 2 + 0.72, lab, ha="center", va="bottom", fontsize=7.5, color=t["ink2"])
        elif x1 - x0 > 2.5:
            arrow((x0, y - bh / 2), (x1, y - bh / 2), 0.35, c)
            ax.text((x0 + x1) / 2, y - bh / 2 - 0.95, lab + " (direct path)", ha="center", va="top",
                    fontsize=7.5, color=t["ink2"])
        else:
            arrow((x0 + bw / 2, y), (x1 - bw / 2, y), 0.0, c)
            ax.text((x0 + x1) / 2, y + bh / 2 + 0.08, lab, ha="center", va="bottom", fontsize=7.2,
                    color=t["ink2"])
    f = d["forecast"]
    ax.text(0.2, 4.45, d["title"], fontsize=10.5, fontweight="bold", color=t["ink"], va="top")
    ax.text(0.2, 0.12, f"{f['n_neurons']:,} neurons, {f['n_synapses']:,} synapses; k = in-degree per "
            f"target neuron. Orange: plastic projections. Engram allocation (Cai 2016) acts on CA3, "
            f"prioritised replay on the store.", fontsize=7.5, color=t["muted"])
    return fig


FIGURES = {"capacity-arms": ("capacity", capacity), "capacity-ratio": ("ratio", ratio),
           "fly-gate": ("gate", gate), "hippocampal-loop": ("loop", loop)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="also write a light PDF of each figure into this directory")
    args = ap.parse_args()
    with open(os.path.join(HERE, "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    for sub in ("png", "svg"):
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    if args.pdf:
        os.makedirs(args.pdf, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "brainsim"
    plt.rcParams["svg.fonttype"] = "none"
    for name, (key, draw) in FIGURES.items():
        for mode, t in THEMES.items():
            style(t)
            fig = draw(data[key], t)
            fig.savefig(os.path.join(HERE, "png", f"{name}-{mode}.png"), dpi=200, bbox_inches="tight")
            if mode == "light":
                fig.savefig(os.path.join(HERE, "svg", f"{name}.svg"), bbox_inches="tight",
                            metadata={"Date": None})
                if args.pdf:
                    fig.savefig(os.path.join(args.pdf, f"{name}.pdf"), bbox_inches="tight",
                                metadata={"CreationDate": None})
            plt.close(fig)
    print("wrote", ", ".join(FIGURES))


if __name__ == "__main__":
    main()
