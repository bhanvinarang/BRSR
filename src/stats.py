"""Stage 6: confidence intervals, McNemar tests, per-field accuracy, k sweep and energy tables.

    python src/stats.py            (from the BRSR folder; needs results/predictions_*.jsonl and results/energy/)

Writes results/stats/*.csv and results/stats/stats_summary.md.

Conventions
  * Accuracy is computed over the 279 accuracy-eligible questions (annotations.acc_main_eligible), FAR over the
    77 questions whose gold answer is 'not disclosed' and that are far_eligible, UAR over the numeric answers
    each system gave.  The comparisons use the same rows for every system, so tests are paired.
  * Bootstrap: 1000 resamples of questions (seed 42), percentile 95% intervals.  Because the 16 questions of a
    company are not independent, a company-level (cluster) bootstrap is reported as a sensitivity check.
  * McNemar: exact two-sided binomial test on the discordant pairs; Holm correction over the three
    accuracy comparisons.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import _flag, evaluate_jsonl, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "stats"
SEED, B = 42, 1000
SYSTEMS = ["S1", "S2", "S3"]
LABEL = {"S1": "S1 RAG zero-shot", "S2": "S2 RAG few-shot", "S3": "S3 RAG + QLoRA"}


def load(annotations, system, k):
    rows = evaluate_jsonl(ROOT / "results" / f"predictions_{system}_k{k}.jsonl", annotations, system=system)
    df = pd.DataFrame(rows).set_index("row_id").sort_index()
    df["acc_row"] = df["acc_main_eligible"].map(_flag)
    df["far_row"] = df["far_eligible"].map(_flag)
    df["uar_row"] = df["grounded"].notna()
    df["ungrounded"] = df["grounded"].map(lambda v: (not v) if v is not None and v == v else np.nan)
    return df


def boot(values, rng, idx=None):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(B, len(values))) if idx is None else idx
    stats = values[idx].mean(axis=1)
    return values.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def cluster_boot(df, column, mask, rng):
    sub = df[mask]
    groups = [g[column].astype(float).values for _, g in sub.groupby("company_id")]
    n = len(groups)
    out = []
    for _ in range(B):
        pick = rng.integers(0, n, size=n)
        out.append(np.concatenate([groups[i] for i in pick]).mean())
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def mcnemar_exact(a, b):
    """a, b: boolean arrays (event happened under system A / B).  Returns (only_a, only_b, p)."""
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * tail)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [None] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def fmt(value, lo=None, hi=None, pct=True):
    scale = 100 if pct else 1
    if lo is None:
        return f"{scale * value:.1f}" if pct else f"{value:.4f}"
    return f"{scale * value:.1f} [{scale * lo:.1f}, {scale * hi:.1f}]"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    frames = {s: load(annotations, s, 5) for s in SYSTEMS}
    ids = frames["S1"].index
    for s in SYSTEMS:
        assert (frames[s].index == ids).all(), "systems must cover the same questions"
    rng = np.random.default_rng(SEED)
    md = ["# Stage 6 results", ""]

    # ---- Table II with CIs
    acc_mask = frames["S1"]["acc_row"].values
    far_mask = frames["S1"]["far_row"].values
    n_acc, n_far = int(acc_mask.sum()), int(far_mask.sum())
    acc_idx = rng.integers(0, n_acc, size=(B, n_acc))        # shared across systems -> paired differences
    far_idx = rng.integers(0, n_far, size=(B, n_far))
    table = []
    for s in SYSTEMS:
        df = frames[s]
        row = {"system": s, "n_accuracy": n_acc, "n_far": n_far}
        for name, col, mask, idx in (("accuracy", "correct", acc_mask, acc_idx), ("far", "false_answer", far_mask, far_idx)):
            m, lo, hi = boot(df.loc[mask, col].astype(float).values, rng, idx)
            clo, chi = cluster_boot(df, col, mask, rng)
            row.update({name: m, name + "_lo": lo, name + "_hi": hi, name + "_cluster_lo": clo, name + "_cluster_hi": chi})
        uar_mask = df["uar_row"].values
        values = df.loc[uar_mask, "ungrounded"].astype(float).values
        m, lo, hi = boot(values, rng)
        row.update({"n_uar": int(uar_mask.sum()), "uar": m, "uar_lo": lo, "uar_hi": hi})
        summary = summarize(df.reset_index().to_dict("records"))
        row["accuracy_answerable"] = summary["accuracy_answerable"]
        row["accuracy_not_disclosed"] = summary["accuracy_not_disclosed"]
        row["format_failure_rate"] = summary["format_failure_rate"]
        table.append(row)
    pd.DataFrame(table).to_csv(OUT / "table2_accuracy_far_uar.csv", index=False)
    md += ["## Table II: accuracy, FAR, UAR (k=5, 95% bootstrap CI over questions)", "",
           "| System | Accuracy % | FAR % | UAR % | Acc. answerable % | Acc. not-disclosed % |", "|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {LABEL[r['system']]} | {fmt(r['accuracy'], r['accuracy_lo'], r['accuracy_hi'])} | "
                  f"{fmt(r['far'], r['far_lo'], r['far_hi'])} | {fmt(r['uar'], r['uar_lo'], r['uar_hi'])} | "
                  f"{fmt(r['accuracy_answerable'])} | {fmt(r['accuracy_not_disclosed'])} |")
    md += ["", f"Accuracy over n={n_acc} eligible questions, FAR over n={n_far}, UAR over each system's numeric answers "
           f"(S1 {table[0]['n_uar']}, S2 {table[1]['n_uar']}, S3 {table[2]['n_uar']}).", "",
           "Company-level (cluster) bootstrap intervals, accuracy and FAR:", ""]
    for r in table:
        md.append(f"- {r['system']}: accuracy {fmt(r['accuracy'], r['accuracy_cluster_lo'], r['accuracy_cluster_hi'])}, "
                  f"FAR {fmt(r['far'], r['far_cluster_lo'], r['far_cluster_hi'])}")

    # ---- McNemar and paired differences
    pairs = [("S1", "S2"), ("S1", "S3"), ("S2", "S3")]
    tests = []
    for metric, col, mask, idx, positive in (("accuracy (correct)", "correct", acc_mask, acc_idx, True),
                                            ("FAR (false answer)", "false_answer", far_mask, far_idx, False)):
        raw = []
        for a, b in pairs:
            xa = frames[a].loc[mask, col].astype(bool).values
            xb = frames[b].loc[mask, col].astype(bool).values
            only_a, only_b, p = mcnemar_exact(xa, xb)
            diff = xb.astype(float) - xa.astype(float)
            d, lo, hi = boot(diff, rng, idx)
            raw.append(p)
            tests.append({"metric": metric, "A": a, "B": b, "n": int(mask.sum()), "only_A": only_a, "only_B": only_b,
                          "diff_B_minus_A": d, "diff_lo": lo, "diff_hi": hi, "p_raw": p})
        adj = holm(raw)
        for t, a_p in zip(tests[-3:], adj):
            t["p_holm"] = a_p
    pd.DataFrame(tests).to_csv(OUT / "mcnemar.csv", index=False)
    md += ["", "## Paired comparisons (exact McNemar; Holm correction over the three pairs per metric)", "",
           "| Metric | A vs B | n | only A | only B | B - A (pp) [95% CI] | p | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    for t in tests:
        md.append(f"| {t['metric']} | {t['A']} vs {t['B']} | {t['n']} | {t['only_A']} | {t['only_B']} | "
                  f"{100 * t['diff_B_minus_A']:+.1f} [{100 * t['diff_lo']:+.1f}, {100 * t['diff_hi']:+.1f}] | "
                  f"{t['p_raw']:.4f} | {t['p_holm']:.4f} |")
    md += ["", "'only A' counts questions where A had the event (correct / false answer) and B did not.", ""]

    # ---- per-field accuracy
    fields = sorted(frames["S1"].loc[acc_mask, "field_id"].unique())
    names = frames["S1"].groupby("field_id")["field_name"].first().to_dict()
    rows = []
    for f in fields:
        sel = acc_mask & (frames["S1"]["field_id"].values == f)
        entry = {"field_id": f, "field_name": names[f], "n": int(sel.sum())}
        for s in SYSTEMS:
            entry[s] = float(frames[s].loc[sel, "correct"].astype(float).mean())
        rows.append(entry)
    pd.DataFrame(rows).to_csv(OUT / "per_field_accuracy.csv", index=False)
    md += ["## Per-field accuracy (%)", "", "| Field | Name | n | S1 | S2 | S3 |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['field_id']} | {r['field_name']} | {r['n']} | {100 * r['S1']:.0f} | {100 * r['S2']:.0f} | {100 * r['S3']:.0f} |")

    # ---- k sweep (S1)
    md += ["", "## k sweep, S1 only", "", "| k | Accuracy % [CI] | FAR % | UAR % | Wh/query | Latency ms |", "|---|---|---|---|---|---|"]
    sweep = []
    for k in (1, 3, 5):
        path = ROOT / "results" / f"predictions_S1_k{k}.jsonl"
        if not path.exists():
            continue
        df = frames["S1"] if k == 5 else load(annotations, "S1", k)
        am, af = df["acc_row"].values, df["far_row"].values
        if k == 5:                                   # reuse Table II so the same number never has two intervals
            main = table[0]
            a, alo, ahi = main["accuracy"], main["accuracy_lo"], main["accuracy_hi"]
            f_, flo, fhi = main["far"], main["far_lo"], main["far_hi"]
        else:
            a, alo, ahi = boot(df.loc[am, "correct"].astype(float).values, rng)
            f_, flo, fhi = boot(df.loc[af, "false_answer"].astype(float).values, rng)
        u = df.loc[df["uar_row"], "ungrounded"].astype(float).mean()
        energy = pd.read_csv(ROOT / "results" / "energy" / f"energy_summary_k{k}.csv")
        energy = energy[energy.system == "S1"].iloc[0]
        entry = {"k": k, "accuracy": a, "accuracy_lo": alo, "accuracy_hi": ahi, "far": f_, "far_lo": flo, "far_hi": fhi,
                 "uar": u, "wh_per_query": energy.wh_total_mean, "latency_ms": energy.latency_ms_mean,
                 "prompt_tokens": energy.prompt_tokens_mean}
        sweep.append(entry)
        md.append(f"| {k} | {fmt(a, alo, ahi)} | {fmt(f_, flo, fhi)} | {fmt(u)} | {energy.wh_total_mean:.4f} | {energy.latency_ms_mean:.0f} |")
    pd.DataFrame(sweep).to_csv(OUT / "k_sweep_S1.csv", index=False)

    # ---- Table III energy and latency, plus training energy
    summary = pd.read_csv(ROOT / "results" / "energy" / "energy_summary_k5.csv")
    train = []
    for fold in range(4):
        data = pd.read_csv(ROOT / "results" / f"energy_train_fold{fold}.csv").iloc[-1]
        train.append({"fold": fold, "kwh": float(data.energy_consumed), "minutes": float(data.duration) / 60})
    train_df = pd.DataFrame(train)
    train_df.to_csv(OUT / "training_energy.csv", index=False)
    runs = pd.read_csv(ROOT / "results" / "energy" / "energy_runs_k5.csv")
    gpu = runs.gpu.iloc[0]
    md += ["", f"## Table III: inference energy and latency (k=5, {gpu}, fp16, greedy, 3 repeats)", "",
           "| System | Wh/query (CodeCarbon) | Wh/query (GPU only, NVML) | Latency ms/query | Prompt tokens | Answer tokens |",
           "|---|---|---|---|---|---|"]
    t3 = []
    for _, r in summary[summary.k == 5].iterrows():
        md.append(f"| {LABEL[r.system]} | {r.wh_total_mean:.4f} ± {r.wh_total_sd:.4f} | {r.wh_gpu_nvml_mean:.4f} ± {r.wh_gpu_nvml_sd:.4f} | "
                  f"{r.latency_ms_mean:.0f} ± {r.latency_ms_sd:.1f} | {r.prompt_tokens_mean:.0f} | {r.new_tokens_mean:.1f} |")
        t3.append(r.to_dict())
    pd.DataFrame(t3).to_csv(OUT / "table3_energy_latency.csv", index=False)
    total_wh = 1000 * train_df.kwh.sum()
    md += ["", "Training energy (QLoRA, one adapter per fold, CodeCarbon):", "",
           "| Fold | Energy Wh | Minutes |", "|---|---|---|"]
    for _, r in train_df.iterrows():
        md.append(f"| {int(r.fold)} | {1000 * r.kwh:.1f} | {r.minutes:.1f} |")
    md.append(f"| total | {total_wh:.1f} | {train_df.minutes.sum():.1f} |")
    wh = {r["system"]: r["wh_total_mean"] for r in t3}
    md += ["", "Break-even (queries after which S3's lower inference energy pays back all four training runs):"]
    for other in ("S1", "S2"):
        saving = wh[other] - wh["S3"]
        md.append(f"- vs {other}: saving {saving:.4f} Wh/query -> about {total_wh / saving:,.0f} queries" if saving > 0
                  else f"- vs {other}: no inference saving")
    md.append("")
    (OUT / "stats_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
