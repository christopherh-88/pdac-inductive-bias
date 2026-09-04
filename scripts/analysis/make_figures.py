#!/usr/bin/env python
"""Every figure in the paper, built from the analysis outputs — nothing drawn by hand.

  fig1_delta_dice_vs_volume.png   H1: per-case delta-Dice against log lesion volume, with the
                                  fitted slope, its patient-level bootstrap band, and the
                                  frozen volume-tertile cut points marked.
  fig2_stratified_maps.png        Deliverable 2: volume tertile x contrast tertile mean Dice,
                                  one panel per arm, per-cell n and bootstrap CI printed in
                                  the cell (sequential single-hue ramp, shared scale).
  fig3_delta_map.png              The same grid as a paired delta, on a diverging scale with a
                                  neutral midpoint, so "no difference" reads as nothing.
  fig4_occlusion_curves.png       Deliverable 3: Dice loss per occlusion shell per arm with
                                  bootstrap CIs, and the CNN effective receptive field marked.
  fig5_identity_control_map.png   Deliverable 4: the identity control's map in the identical
                                  layout to fig2, plus the paired difference panel.
  fig6_loso.png                   Deliverables 5 & 6: cross-source Dice drop per source per
                                  arm, with the boundary-tolerance floor drawn across it.

Every figure is skipped, with a printed reason, when its inputs do not exist yet — so this
runs unchanged after Tier A (figs 1-4) and again after Tier B (all six).

Usage:
  python scripts/analysis/make_figures.py --results results --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from figure_style import (AXIS, DIVERGING, INK, INK_SECONDARY, MUTED, SEQUENTIAL,
                          apply_style, arm_color, arm_label, despine)
from metrics import CFG, bootstrap_ci

TERTILE_NAMES = {0: "small", 1: "medium", 2: "large"}
CNR_NAMES = {0: "low", 1: "mid", 2: "high"}


def _save(fig, path: Path, name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _frozen_volume_cuts():
    p = Path(__file__).parents[2] / "config" / "frozen_thresholds.yaml"
    if not p.exists():
        return None
    return (yaml.safe_load(open(p)) or {}).get("strata", {}).get("volume_tertile_cuts_mm3")


# --------------------------------------------------------------------------- fig 1
def fig1_delta_vs_volume(per_case: Path, out: Path):
    df = pd.read_csv(per_case)
    st = CFG["statistics"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.scatter(df["log_volume"], df["delta_dice"], s=22, alpha=0.55, linewidths=0,
               color=arm_color("cnn"), zorder=3)

    # Fitted line + patient-level bootstrap band over the same grid.
    xs = np.linspace(df["log_volume"].min(), df["log_volume"].max(), 60)
    fit = np.polyfit(df["log_volume"], df["delta_dice"], 1)
    rng = np.random.default_rng(st["bootstrap_seed"])
    groups = df.groupby("patient_id").indices
    patients = np.array(list(groups.keys()))
    curves = []
    for _ in range(min(st["bootstrap_resamples"], 500)):
        sample = rng.choice(patients, size=len(patients), replace=True)
        b = df.iloc[np.concatenate([groups[p] for p in sample])]
        try:
            c = np.polyfit(b["log_volume"], b["delta_dice"], 1)
            curves.append(np.polyval(c, xs))
        except Exception:
            pass
    if curves:
        lo, hi = np.percentile(np.vstack(curves), [2.5, 97.5], axis=0)
        ax.fill_between(xs, lo, hi, color=arm_color("cnn"), alpha=0.15, linewidth=0, zorder=2)
    ax.plot(xs, np.polyval(fit, xs), color=arm_color("cnn"), linewidth=2.0, zorder=4)

    cuts = _frozen_volume_cuts()
    if cuts:
        lo_c, hi_c = np.log(cuts[0]), np.log(cuts[1])
        for c in (lo_c, hi_c):
            ax.axvline(c, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
        # Tertile names sit *inside* the axes, centred on each band's visible span — above the
        # axes they land in the title, and a fixed offset from the cut points can fall off the
        # end of the data range entirely.
        x0, x1 = ax.get_xlim()
        for name, x in zip(("small", "medium", "large"),
                           ((x0 + lo_c) / 2, (lo_c + hi_c) / 2, (hi_c + x1) / 2)):
            ax.text(x, 0.985, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.5, color=MUTED)

    ax.set_xlabel("log lesion volume (mm³)")
    ax.set_ylabel("ΔDice  (CNN − transformer)")
    ax.set_title("Per-case CNN advantage against lesion volume\n"
                 "line: OLS fit · band: 95% patient-level bootstrap · dashed: frozen tertile cuts",
                 loc="left", color=INK)
    despine(ax)
    _save(fig, out / "fig1_delta_dice_vs_volume.png", "fig1")


# --------------------------------------------------------------------------- fig 2 / 5
def _ink_on(rgba):
    """Pick label ink by the cell's actual rendered luminance, not by a guess about where the
    value sits in the range — the two diverge for any non-linear ramp, and an unreadable label
    on one cell of a nine-cell map is exactly the kind of thing nobody notices until print."""
    r, g, b = rgba[:3]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if luminance < 0.45 else INK


def _draw_map(ax, grid, n_grid, cmap, vmin, vmax, title, fmt="{:.3f}", ci_grid=None,
              show_ylabel=True):
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([CNR_NAMES.get(i, i) for i in range(grid.shape[1])])
    ax.set_yticklabels([TERTILE_NAMES.get(i, i) for i in range(grid.shape[0])])
    ax.set_xlabel("contrast tertile (CNR)")
    # Only the leftmost panel is labelled: repeating an identical axis label on every panel of
    # a shared-axis row is noise, and it collides with the neighbouring panel.
    ax.set_ylabel("lesion volume tertile" if show_ylabel else "")
    ax.set_title(title, loc="left", color=INK, pad=8)
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "n=0", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            # Direct value labels are mandatory relief for sub-3:1 cells, and they make the
            # map readable in greyscale print regardless.
            label = fmt.format(v) + f"\nn={int(n_grid[i, j])}"
            if ci_grid is not None and not np.isnan(ci_grid[i, j][0]):
                label += f"\n[{ci_grid[i, j][0]:.2f}, {ci_grid[i, j][1]:.2f}]"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.8,
                    color=_ink_on(im.cmap(im.norm(v))), linespacing=1.25)
    despine(ax, keep=())
    return im


def _grids(cells: pd.DataFrame, value_col: str, arm: str = None):
    sub = cells[cells["arm"] == arm] if arm is not None else cells
    n_v = int(cells["volume_tertile"].max()) + 1
    n_c = int(cells["cnr_tertile"].max()) + 1
    g = np.full((n_v, n_c), np.nan)
    n = np.zeros((n_v, n_c))
    ci = np.full((n_v, n_c, 2), np.nan)
    for r in sub.itertuples():
        i, j = int(r.volume_tertile), int(r.cnr_tertile)
        g[i, j] = getattr(r, value_col)
        n[i, j] = r.n
        if hasattr(r, "ci_lo") and not pd.isna(r.ci_lo):
            ci[i, j] = (r.ci_lo, r.ci_hi)
    return g, n, ci


def fig2_stratified_maps(heatmap_csv: Path, out: Path):
    cells = pd.read_csv(heatmap_csv)
    arms = list(dict.fromkeys(cells["arm"]))
    vals = cells["dice_mean"].dropna()
    vmin, vmax = float(vals.min()), float(vals.max())
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 4.3))
    axes = np.atleast_1d(axes)
    for k, (ax, arm) in enumerate(zip(axes, arms)):
        g, n, ci = _grids(cells, "dice_mean", arm)
        im = _draw_map(ax, g, n, SEQUENTIAL, vmin, vmax, arm_label(arm), ci_grid=ci,
                       show_ylabel=(k == 0))
    fig.colorbar(im, ax=list(axes), shrink=0.82, label="mean Dice")
    fig.suptitle("Stratified failure map: mean Dice by lesion volume × contrast\n"
                 "cell text: mean Dice, n, 95% patient-level bootstrap CI",
                 x=0.02, y=1.0, va="bottom", ha="left", color=INK, fontsize=10)
    fig.savefig(out / "fig2_stratified_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'fig2_stratified_maps.png'}")


def fig3_delta_map(per_case: Path, out: Path):
    df = pd.read_csv(per_case)
    st = CFG["statistics"]
    rows = []
    for (vt, ct), g in df.groupby(["volume_tertile", "cnr_tertile"]):
        lo, hi = bootstrap_ci(g, lambda b: b["delta_dice"].mean(),
                              st["bootstrap_resamples"], st["bootstrap_seed"] + 4)
        rows.append({"volume_tertile": vt, "cnr_tertile": ct, "n": len(g),
                     "delta_dice": g["delta_dice"].mean(), "ci_lo": lo, "ci_hi": hi})
    cells = pd.DataFrame(rows)
    g, n, ci = _grids(cells, "delta_dice")
    lim = float(np.nanmax(np.abs(g))) or 0.01

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = _draw_map(ax, g, n, DIVERGING, -lim, lim,
                   "Paired ΔDice (CNN − transformer) by stratum", fmt="{:+.3f}", ci_grid=ci)
    fig.colorbar(im, ax=ax, shrink=0.82, label="ΔDice (CNN − transformer)")
    _save(fig, out / "fig3_delta_map.png", "fig3")


def fig5_identity_map(h2b_dir: Path, out: Path):
    cells = pd.read_csv(h2b_dir / "identity_vs_full_heatmap.csv")
    summary = yaml.safe_load(open(h2b_dir / "summary.yaml"))
    vals = pd.concat([cells["dice_full"], cells["dice_identity"]]).dropna()
    vmin, vmax = float(vals.min()), float(vals.max())

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    for k, (ax, col, title) in enumerate(
            ((axes[0], "dice_full", arm_label(summary["arm_full"])),
             (axes[1], "dice_identity", arm_label(summary["arm_identity"])))):
        g, n, _ = _grids(cells, col)
        im = _draw_map(ax, g, n, SEQUENTIAL, vmin, vmax, title, show_ylabel=(k == 0))
    fig.colorbar(im, ax=[axes[0], axes[1]], shrink=0.82, label="mean Dice")

    g, n, ci = _grids(cells, "delta_dice")
    lim = float(np.nanmax(np.abs(g))) or 0.01
    im2 = _draw_map(axes[2], g, n, DIVERGING, -lim, lim,
                    "Paired difference (full − identity)", fmt="{:+.3f}", ci_grid=ci,
                    show_ylabel=False)
    fig.colorbar(im2, ax=axes[2], shrink=0.82, label="ΔDice")

    verdict = ("map reproduced within CI in every cell"
               if summary["map_reproduced_within_ci"] else
               f"{summary['n_cells_agreeing_within_ci']}/{summary['n_cells']} cells agree "
               "within CI")
    fig.suptitle("H2b identity control, laid out identically to the full transformer's map — "
                 + verdict, x=0.02, y=1.0, va="bottom", ha="left", color=INK, fontsize=10)
    fig.savefig(out / "fig5_identity_control_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / 'fig5_identity_control_map.png'}")


# --------------------------------------------------------------------------- fig 4
def fig4_occlusion(occ_csv: Path, out: Path, erf_yaml: Path = None):
    df = pd.read_csv(occ_csv)
    st = CFG["statistics"]
    shells = [tuple(s) for s in CFG["hypotheses"]["h2"]["occlusion_shells_mm"]]
    centers = [(lo + hi) / 2 for lo, hi in shells]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for arm in dict.fromkeys(df["arm"]):
        means, los, his = [], [], []
        for lo_mm, hi_mm in shells:
            g = df[(df["arm"] == arm) & (df["shell"] == f"{lo_mm}-{hi_mm}")]
            if g.empty:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            m = g["dice_loss"].mean()
            l, h = bootstrap_ci(g, lambda b: b["dice_loss"].mean(),
                                st["bootstrap_resamples"], st["bootstrap_seed"])
            means.append(m); los.append(l); his.append(h)
        c = arm_color(arm)
        ax.fill_between(centers, los, his, color=c, alpha=0.15, linewidth=0)
        ax.plot(centers, means, color=c, marker="o", markersize=6, label=arm_label(arm))
        # Direct label at the last point, in ink rather than the series colour.
        if not np.isnan(means[-1]):
            ax.annotate(arm_label(arm), (centers[-1], means[-1]), textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=8, color=INK_SECONDARY)

    if erf_yaml and Path(erf_yaml).exists():
        erf = yaml.safe_load(open(erf_yaml))
        r95 = erf.get("erf_radius_mm_95pct")
        if r95:
            ax.axvline(r95, color=MUTED, linestyle=(0, (5, 3)), linewidth=1.2)
            ax.annotate(f"CNN effective receptive field\n95% gradient mass: {r95:.0f} mm",
                        (r95, ax.get_ylim()[1]), textcoords="offset points", xytext=(6, -14),
                        fontsize=7.5, color=MUTED)

    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in shells])
    ax.set_xlabel("occluded shell, distance from lesion surface (mm)")
    ax.set_ylabel("Dice loss vs unoccluded prediction")
    ax.set_title("H2 occlusion test: how far from the lesion each arm reads\n"
                 "band: 95% patient-level bootstrap CI", loc="left", color=INK)
    ax.legend(loc="upper left")
    ax.set_xlim(centers[0] - 8, centers[-1] + 26)
    despine(ax)
    _save(fig, out / "fig4_occlusion_curves.png", "fig4")


# --------------------------------------------------------------------------- fig 6
def fig6_loso(by_source_csv: Path, out: Path, floor_csv: Path = None):
    bys = pd.read_csv(by_source_csv)
    sources = list(dict.fromkeys(bys["held_out_source"]))
    arms = list(dict.fromkeys(bys["arm"]))
    x = np.arange(len(sources), dtype=float)
    # 2px-equivalent gap between adjacent bars: width leaves a visible surface strip.
    width = 0.78 / max(len(arms), 1) * 0.92

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, arm in enumerate(arms):
        g = bys[bys["arm"] == arm].set_index("held_out_source").reindex(sources)
        offs = x + (k - (len(arms) - 1) / 2) * (width / 0.92)
        err = np.vstack([g["dice_drop"] - g["dice_drop_ci_lo"],
                         g["dice_drop_ci_hi"] - g["dice_drop"]])
        ax.bar(offs, g["dice_drop"], width=width, color=arm_color(arm), label=arm_label(arm),
               linewidth=0)
        ax.errorbar(offs, g["dice_drop"], yerr=np.abs(err), fmt="none", ecolor=INK_SECONDARY,
                    elinewidth=1.0, capsize=3)
        # Anchor the value label above the CI whisker, not the bar top, so the two never collide.
        for xi, v, lo_i, hi_i in zip(offs, g["dice_drop"], g["dice_drop_ci_lo"],
                                     g["dice_drop_ci_hi"]):
            if not np.isnan(v):
                anchor = max(v, hi_i) if v >= 0 else min(v, lo_i)
                ax.annotate(f"{v:+.3f}", (xi, anchor), textcoords="offset points",
                            xytext=(0, 3 if v >= 0 else -3), ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=7, color=INK_SECONDARY)

    # Headroom first, so the bar value labels, the legend, and the floor annotation are not
    # all competing for the same strip of the axes.
    top = max(np.nanmax(bys["dice_drop_ci_hi"].to_numpy(dtype=float)), 0.0)
    if floor_csv and Path(floor_csv).exists():
        floor = pd.read_csv(floor_csv)
        tolerable = float((1 - floor["tolerance_floor_dice"]).median())
        ax.set_ylim(top=max(top, tolerable) * 1.38)
        # Named in the legend rather than annotated on the plot: an in-plot label for a
        # horizontal reference line has nowhere safe to sit once the bars are tall.
        ax.axhline(tolerable, color=MUTED, linestyle=(0, (5, 3)), linewidth=1.2,
                   label=f"boundary-tolerance floor (median tolerable Dice loss {tolerable:.3f})")
    else:
        ax.set_ylim(top=top * 1.25)

    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_xlabel("held-out source")
    ax.set_ylabel("Dice loss (in-domain − leave-one-source-out)")
    ax.set_title("H3 cross-source degradation against the boundary-tolerance floor\n"
                 "error bars: 95% patient-level bootstrap CI", loc="left", color=INK)
    ax.legend(loc="upper left")
    despine(ax)
    _save(fig, out / "fig6_loso.png", "fig6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=Path("results"), type=Path,
                    help="root holding h1/, h2/, h2b/, h3/, loso/ from the analysis scripts")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    out = args.out or (args.results / "figures")
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    R = args.results

    jobs = [
        ("fig1", lambda: fig1_delta_vs_volume(R / "h1" / "per_case.csv", out),
         [R / "h1" / "per_case.csv"]),
        ("fig2", lambda: fig2_stratified_maps(R / "h1" / "stratum_heatmap.csv", out),
         [R / "h1" / "stratum_heatmap.csv"]),
        ("fig3", lambda: fig3_delta_map(R / "h1" / "per_case.csv", out),
         [R / "h1" / "per_case.csv"]),
        ("fig4", lambda: fig4_occlusion(R / "h2" / "occlusion_per_case.csv", out,
                                        R / "erf_cnn.yaml"),
         [R / "h2" / "occlusion_per_case.csv"]),
        ("fig5", lambda: fig5_identity_map(R / "h2b", out),
         [R / "h2b" / "identity_vs_full_heatmap.csv", R / "h2b" / "summary.yaml"]),
        ("fig6", lambda: fig6_loso(R / "loso" / "by_source.csv", out,
                                   R / "h3" / "boundary_floor_per_case.csv"),
         [R / "loso" / "by_source.csv"]),
    ]
    made = 0
    for name, fn, inputs in jobs:
        missing = [str(p) for p in inputs if not Path(p).exists()]
        if missing:
            print(f"{name}: SKIP — missing {missing}")
            continue
        print(f"{name}:")
        fn()
        made += 1
    print(f"\n{made}/{len(jobs)} figures written to {out}/")


if __name__ == "__main__":
    main()
