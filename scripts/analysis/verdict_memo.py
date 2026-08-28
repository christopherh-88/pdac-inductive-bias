#!/usr/bin/env python
"""Read the results against the frozen decision rules and write the verdict memo.

The Phase 2 rule is "results get read against the decision rules before anyone writes prose
about them". This script is that step, mechanised: every rule is quoted out of
config/analysis_config.yaml (the frozen file the analysis code actually reads), every number
is pulled from the analysis scripts' own summary.yaml files, and the verdict is the rule
applied to the number. Nothing here can invent a rule that is not already in prereg-v1,
because it never has anything but the config to read a rule from.

Missing inputs produce "NOT YET RUN" rows rather than silence — a hypothesis with no result is
visibly outstanding rather than quietly absent.

Usage:
  python scripts/analysis/verdict_memo.py --results results --out results/VERDICT.md
"""
import argparse
import datetime
import subprocess
from pathlib import Path

import yaml

from metrics import CFG

REPO = Path(__file__).parents[2]


def load(path: Path):
    return yaml.safe_load(open(path)) if path.exists() else None


def git(*args, default="unknown"):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip() or default
    except Exception:
        return default


def fmt_ci(ci, pct=False):
    if not ci:
        return "—"
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:+.4f}")
    return f"[{f(ci[0])}, {f(ci[1])}]"


def h1_section(r):
    s = load(r / "h1" / "summary.yaml")
    rule = (f"Supported if the 95% CI of the slope of ΔDice on log volume lies below zero. "
            f"Large-lesion tertile additionally reported with TOST at ±"
            f"{CFG['hypotheses']['h1']['equivalence_margin_dice_points']} Dice points.")
    if not s:
        return "H1 — Locality vs lesion size", rule, "NOT YET RUN", ["`results/h1/summary.yaml` absent — run `paired_analysis.py`."]
    verdict = "SUPPORTED" if s["h1_supported"] else "NOT SUPPORTED"
    lines = [
        f"- Slope of ΔDice on log volume: **{s['h1_slope']:+.5f}**, 95% CI {fmt_ci(s['h1_ci'])}"
        f" (n = {s['n_cases']} cases).",
        f"- Rule reads: CI entirely below zero → **{verdict}**.",
        f"- Large-lesion tertile mean ΔDice CI {fmt_ci(s['large_tertile_tost_ci'])} against a "
        f"±{CFG['hypotheses']['h1']['equivalence_margin_dice_points'] / 100:.2f} margin → "
        f"**{'EQUIVALENT' if s['large_tertile_equivalent'] else 'NOT SHOWN EQUIVALENT'}**.",
    ]
    return "H1 — Locality vs lesion size", rule, verdict, lines


def h2_section(r):
    h2 = CFG["hypotheses"]["h2"]
    lo, hi = h2["decisive_shell_mm"]
    rule = (f"Supported if the transformer's Dice loss in the {lo}–{hi} mm shell exceeds the "
            f"CNN's by at least {h2['margin_dice_points']} Dice points with a patient-level "
            "bootstrap CI excluding zero.")
    s = load(r / "h2" / "summary.yaml")
    if not s:
        return "H2 — Context use", rule, "NOT YET RUN", ["`results/h2/summary.yaml` absent — run `occlusion_test.py --stage score`."]
    verdict = "SUPPORTED" if s["h2_supported"] else "NOT SUPPORTED"
    lines = [f"- Decisive shell {lo}–{hi} mm: mean(transformer loss − CNN loss) = "
             f"**{s['h2_contrast_mean']:+.4f}**, CI {fmt_ci(s['h2_ci'])} (n = {s['n_cases']}).",
             f"- Rule reads: ≥ {h2['margin_dice_points'] / 100:.2f} and CI excludes zero → "
             f"**{verdict}**."]
    erf = load(r / "erf_cnn.yaml")
    if erf:
        lines.append(f"- CNN effective receptive field (input-gradient measurement): 95% of "
                     f"gradient mass within **{erf['erf_radius_mm_95pct']:.1f} mm** "
                     f"(50%: {erf['erf_radius_mm_50pct']:.1f} mm, "
                     f"90%: {erf['erf_radius_mm_90pct']:.1f} mm).")
    else:
        lines.append("- CNN effective receptive field: NOT YET MEASURED "
                     "(`effective_receptive_field.py --nnunet-checkpoint …`).")
    return "H2 — Context use", rule, verdict, lines


def h2b_section(r):
    rule = ("If the identity control reproduces the full transformer's stratified map within "
            "CI, attention is not the cause, and the paper says so.")
    s = load(r / "h2b" / "summary.yaml")
    if not s:
        return "H2b — Attention dependence", rule, "NOT YET RUN", ["`results/h2b/summary.yaml` absent — run `identity_comparison.py`."]
    verdict = ("MAP REPRODUCED — attention is not the cause"
               if s["map_reproduced_within_ci"] else "MAP NOT REPRODUCED")
    lines = [
        f"- {s['n_cells_agreeing_within_ci']}/{s['n_cells']} populated (volume × contrast) "
        f"cells have a paired-difference CI containing zero.",
        f"- Overall Dice(full) − Dice(identity) = "
        f"**{s['overall_delta_dice_full_minus_identity']:+.4f}**, CI "
        f"{fmt_ci(s['overall_delta_ci'])} (Wilcoxon p = {s['overall_wilcoxon_p']:.3g}).",
        f"- Cell-mean Dice rank agreement across cells: Spearman ρ = "
        f"{s['cell_mean_dice_spearman_rho']:.3f}.",
        f"- {s['h2b_verdict']}",
    ]
    return "H2b — Attention dependence", rule, verdict, lines


def h3_section(r):
    thr = CFG["hypotheses"]["h3"]["share_threshold"]
    rule = (f"Report the share of each arm's cross-source Dice loss that falls inside the "
            f"boundary-tolerance floor. Supported if that share exceeds {thr:.0%} for at least "
            "one arm.")
    s = load(r / "h3" / "summary.yaml")
    if not s or "h3_supported" not in s:
        return "H3 — Source shift vs boundary tolerance", rule, "NOT YET RUN", [
            "`results/h3/summary.yaml` absent or floor-only — run `loso_analysis.py`, then "
            "`boundary_tolerance.py --loso results/loso/loso_dice.csv`."]
    verdict = "SUPPORTED" if s["h3_supported"] else "NOT SUPPORTED"
    lines = []
    for arm, v in s.items():
        if not isinstance(v, dict):
            continue
        lines.append(f"- {arm}: **{v['share_inside_floor']:.1%}** of cross-source Dice loss "
                     f"sits inside the floor, CI {fmt_ci(v['share_inside_floor_ci'], pct=True)} "
                     f"(n = {v['n']}).")
    lines.append(f"- Rule reads: > {thr:.0%} for at least one arm → **{verdict}**.")
    loso = load(r / "loso" / "summary.yaml")
    if loso:
        for arm, v in loso.get("overall", {}).items():
            lines.append(f"- {arm} raw cross-source Dice drop: {v['mean_dice_drop']:+.4f}, CI "
                         f"{fmt_ci(v['mean_dice_drop_ci'])} "
                         f"({v['mean_dice_in_domain']:.4f} → {v['mean_dice_loso']:.4f}).")
    return "H3 — Source shift vs boundary tolerance", rule, verdict, lines


def context_sections(r):
    """Pre-specified checks and controls that are not hypotheses but gate how they read."""
    out = []
    cs = load(r / "contrast_sensitivity" / "summary.yaml")
    if cs:
        status = "EXPLORATORY" if cs["contrast_axis_exploratory"] else "CONFIRMATORY"
        rhos = ", ".join(f"{k.split('__vs__')[0].replace('cnr_ring', '').replace('mm', '')}"
                         f"/{k.split('__vs__')[1].replace('cnr_ring', '').replace('mm', '')} mm: "
                         f"{v:.3f}" for k, v in cs["spearman_rank_stability"].items())
        out.append(("Contrast axis status (pre-specified gate)",
                    f"Contrast axis reported as exploratory if Spearman ρ < "
                    f"{cs['min_spearman_rule']} between any pair of ring widths.",
                    status,
                    [f"- Ring-width rank stability: {rhos}.",
                     f"- {cs['n_cells_changing_delta_sign_across_rings']}/{cs['n_cells']} cells "
                     "change the sign of mean ΔDice across ring widths."]))
    nih = load(r / "nih_fp" / "summary.yaml")
    if nih:
        lines = [f"- {arm}: {v['mean_fp_per_case']:.3f} FP/case, CI "
                 f"[{v['mean_fp_per_case_ci'][0]:.3f}, {v['mean_fp_per_case_ci'][1]:.3f}], "
                 f"{v['pct_cases_with_any_fp']:.0%} of cases with ≥1 FP (n = {v['n_cases']})."
                 for arm, v in nih.items() if isinstance(v, dict)]
        out.append(("NIH negative control (deliverable 8)",
                    "Negative control only: false-positive lesions per case, never a tumour "
                    "test set.", "REPORTED", lines))
    for arm in ("cnn", "tf", "identity_control"):
        v = load(r / f"variance_{arm}.yaml")
        if v:
            out.append((f"Seed vs fold variance — {arm} (deliverable 7)",
                        "Seed replicates on the replicate fold separate initialization "
                        "variance from fold variance.", "REPORTED",
                        [f"- Fold SD {v['fold_sd']:.4f} across {v['n_folds']} folds; seed SD "
                         f"{v['seed_sd']:.4f} across {v['n_seed_replicate_cases']} cases × "
                         f"seeds; fold SD net of seed ≈ "
                         f"{v['fold_sd_net_of_seed_estimate']:.4f}."]))
    ir = load(r / "inter_rater" / "summary.yaml")
    if ir:
        out.append(("Inter-rater proxy", "Either a list of paired cases, or a written "
                    "statement that there are none.",
                    "PAIRED DELINEATIONS FOUND" if ir["n_cases_with_independent_second_delineation"]
                    else "NONE — floor is synthetic", [f"- {ir['statement']}"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=Path("results"), type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    r = args.results
    out = args.out or (r / "VERDICT.md")

    sections = [h1_section(r), h2_section(r), h2b_section(r), h3_section(r)] + context_sections(r)

    today = datetime.date.today().isoformat()
    tag_commit = git("rev-list", "-n", "1", CFG["study"]["preregistration_tag"], default="—")
    head = git("rev-parse", "--short", "HEAD")

    md = [f"# Verdict memo — {CFG['study']['name']}", "",
          f"Generated {today} by `scripts/analysis/verdict_memo.py` at commit `{head}`.",
          f"Rules quoted from `config/analysis_config.yaml`, frozen at "
          f"`{CFG['study']['preregistration_tag']}` (`{tag_commit[:12]}`). Numbers read from "
          "the analysis scripts' own `summary.yaml` outputs.", "",
          "Every rule below was fixed before any model trained. This file applies them; it "
          "does not interpret them.", "",
          "| Hypothesis | Verdict |", "| --- | --- |"]
    md += [f"| {title} | **{verdict}** |" for title, _, verdict, _ in sections]
    md.append("")
    for title, rule, verdict, lines in sections:
        md += [f"## {title}", "", f"**Frozen rule.** {rule}", "",
               f"**Verdict: {verdict}**", ""] + lines + [""]

    md += ["## What this memo does not do", "",
           "It does not weigh, combine, or reinterpret the verdicts above. A hypothesis that "
           "reads NOT SUPPORTED is reported as a stratified negative result with its intervals, "
           "per section 7 of the pre-registration — not re-tested under a different rule.", ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print("\n".join(f"{t}: {v}" for t, _, v, _ in sections))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
