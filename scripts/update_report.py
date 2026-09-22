"""
update_report.py — generate the report's numbers as LaTeX macros and tables.

`report/generated.tex` is written from `results/metrics/unified_results.json` and
`ablation.json`, and `main.tex` uses the macros rather than typing figures into
prose. This is the same discipline as `update_readme.py`, applied to the report,
and for the same reason: the previous `main.tex` carried a *third* set of numbers
that matched neither the README nor the results files — including a row that
listed YOLOv8's MAE as 8.47, which was Watershed's value pasted into the wrong
cell, and a headline "62.3% MAE reduction" computed from it.

    .venv/bin/python scripts/update_report.py
    .venv/bin/python scripts/update_report.py --check    # fail if stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import COUNT_TOLERANCE, METRICS_DIR, ROOT

OUT = ROOT / "report" / "generated.tex"

# LaTeX command names may only contain letters.
NAMES = {"watershed": "Ws", "kmeans": "Km", "yolo": "Yo", "hybrid": "Hy"}
ROWS = [
    ("watershed", r"Watershed", "Classical"),
    ("kmeans", r"KMeans ($k{=}5$)", "Classical"),
    ("yolo", r"YOLOv8s-seg (fine-tuned)", "Deep learning"),
    ("hybrid", r"\textbf{Hybrid (density-aware)}", "Hybrid"),
]


def render(res: dict, abl: dict) -> str:
    s, p = res["summary"], res["provenance"]
    a = abl["attribution"]
    t = p.get("training") or {}

    L: list[str] = [
        "% ─────────────────────────────────────────────────────────────────────",
        "% GENERATED FILE — do not edit.",
        "% Written by scripts/update_report.py from results/metrics/*.json.",
        "% Every number in main.tex comes from here, so the report cannot drift",
        "% from the run that produced it.",
        "% ─────────────────────────────────────────────────────────────────────",
        "",
        f"\\newcommand{{\\runDate}}{{{p['date']}}}",
        f"\\newcommand{{\\runSeed}}{{{p['seed']}}}",
        f"\\newcommand{{\\runDevice}}{{{p['device']}}}",
        f"\\newcommand{{\\nTest}}{{{p['n_test_images']}}}",
        f"\\newcommand{{\\countTol}}{{{COUNT_TOLERANCE}}}",
        f"\\newcommand{{\\trainEpochs}}{{{t.get('epochs', '--')}}}",
        f"\\newcommand{{\\trainImgsz}}{{{t.get('imgsz', '--')}}}",
        f"\\newcommand{{\\trainBatch}}{{{t.get('batch', '--')}}}",
        "",
    ]

    for key, cmd in NAMES.items():
        m = s[key]
        L += [
            f"\\newcommand{{\\{cmd}IoU}}{{{m['mean_iou']:.3f}}}",
            f"\\newcommand{{\\{cmd}Acc}}{{{m['count_accuracy_pct']:.0f}}}",
            f"\\newcommand{{\\{cmd}Mae}}{{{m['mae']:.2f}}}",
            f"\\newcommand{{\\{cmd}Ms}}{{{m['median_inference_ms']:.0f}}}",
            f"\\newcommand{{\\{cmd}Pred}}{{{m['avg_pred_count']:.1f}}}",
        ]
    L += [
        f"\\newcommand{{\\gtCount}}{{{s['yolo']['avg_gt_count']:.1f}}}",
        f"\\newcommand{{\\gateFired}}{{{s['hybrid']['dense_gate_fired']}}}",
        f"\\newcommand{{\\blendApplied}}{{{s['hybrid']['watershed_blend_applied']}}}",
        f"\\newcommand{{\\gainTotal}}{{{a['total_objects_recovered']}}}",
        f"\\newcommand{{\\gainRelaxed}}{{{a['from_relaxed_detector_pass']}}}",
        f"\\newcommand{{\\gainWatershed}}{{{a['from_watershed_cross_check']}}}",
        f"\\newcommand{{\\watershedShare}}{{{a['watershed_share_pct']:.0f}}}",
        f"\\newcommand{{\\accGain}}{{{s['hybrid']['count_accuracy_pct'] - s['yolo']['count_accuracy_pct']:.0f}}}",
        f"\\newcommand{{\\maeDrop}}{{{100 * (s['yolo']['mae'] - s['hybrid']['mae']) / s['yolo']['mae']:.0f}}}",
        "",
    ]

    # ── Main results table ────────────────────────────────────────────────────
    L += [
        r"\newcommand{\resultsTable}{%",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{All four methods evaluated in a single run on the same "
        rf"{p['n_test_images']} held-out test images, with identical metric "
        rf"definitions, on one device (\texttt{{{p['device']}}}, seed "
        rf"{p['seed']}, \runDate). Instance IoU is class-agnostic under greedy "
        r"one-to-one matching; unmatched ground-truth instances score zero and "
        r"are included in the mean. Latency is a median with an interquartile "
        r"range.}",
        r"\label{tab:results}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        rf"Method & Type & Mean IoU & Count acc.\ ($\pm${COUNT_TOLERANCE}) & MAE & Median (ms) \\",
        r"\midrule",
    ]
    for key, name, typ in ROWS:
        m = s[key]
        bold = key == "hybrid"

        def b(x, _bold=bold):
            return r"\textbf{" + x + "}" if _bold else x

        iou = b("{:.3f}".format(m["mean_iou"]))
        acc = b("{:.0f}".format(m["count_accuracy_pct"]) + r"\%")
        mae = b("{:.2f}".format(m["mae"]))
        ms = "{:.0f}".format(m["median_inference_ms"])
        L.append(f"{name} & {typ} & {iou} & {acc} & {mae} & {ms} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"}", ""]

    # ── Ablation table ────────────────────────────────────────────────────────
    gf, gq = abl["gate_fired"], abl["gate_did_not_fire"]
    L += [
        r"\newcommand{\ablationTable}{%",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Decomposition of the hybrid's improvement over YOLOv8s-seg "
        r"alone. The density-gated relaxed detector pass accounts for almost all "
        r"of it; the watershed cross-check contributes \watershedShare\% while "
        r"firing on \blendApplied\ of \nTest\ images.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Subset & Images & MAE (YOLO $\rightarrow$ hybrid) & Objects recovered \\",
        r"\midrule",
        rf"Gate fired & {gf['n']} & {gf['yolo_mae']:.2f} $\rightarrow$ "
        rf"{gf['hybrid_mae']:.2f} & {gf['objects_recovered']} \\",
        rf"Gate did not fire & {gq['n']} & {gq['yolo_mae']:.2f} $\rightarrow$ "
        rf"{gq['hybrid_mae']:.2f} & {gq['objects_recovered']} \\",
        r"\midrule",
        rf"\textit{{of which:}} relaxed pass & -- & -- & \gainRelaxed \\",
        rf"\textit{{of which:}} watershed & \blendApplied & -- & \gainWatershed \\",
        r"\midrule",
        rf"\textbf{{Total}} & \textbf{{\nTest}} & \textbf{{\YoMae\ $\rightarrow$ "
        rf"\HyMae}} & \textbf{{\gainTotal}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"}",
        "",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    check = "--check" in sys.argv
    res_p = METRICS_DIR / "unified_results.json"
    abl_p = METRICS_DIR / "ablation.json"
    for p_ in (res_p, abl_p):
        if not p_.exists():
            sys.exit(f"Missing {p_}. Run src/evaluate.py and src/ablation.py first.")

    text = render(json.loads(res_p.read_text()), json.loads(abl_p.read_text()))

    if check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit("report/generated.tex is STALE — run without --check.")
        print("report/generated.tex is up to date.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
