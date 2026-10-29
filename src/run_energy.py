"""Stage 5: energy and latency per query for S1 / S2 / S3 (same GPU, same session, same precision).

    python src/run_energy.py                          # 3 systems x 3 repeats x 320 questions
    python src/run_energy.py --per-fold 20            # cheaper trial: 80 questions per repeat
    python src/run_energy.py --systems S1 --k 1       # k sweep for Stage 6 (writes its own CSV)
    python src/run_energy.py --summarize-only         # rebuild the summary table from the CSV
    python src/run_energy.py --dry-run                # stub model, no GPU (checks the plumbing only)

Method
  * One CodeCarbon tracker wraps only the generation loop of one (system, repeat, fold) block;
    model loading and a short warm-up (--warmup questions, not tracked) happen before it starts.
  * Repeats are the outer loop and the system order rotates (S1,S2,S3 / S2,S3,S1 / S3,S1,S2) so slow
    GPU drift (heat, clocks) does not always fall on the same system.
  * S3 loads each fold's merged adapter, so it runs 4 blocks per repeat; the other systems share one
    base model.  Energy of the 4 blocks is summed, so every system is charged for all 320 questions.
  * Besides CodeCarbon (GPU + CPU + RAM, CPU/RAM partly estimated) the GPU's own energy counter
    (NVML, joules) is read around the same loop as a cross-check.
  * Decoding is greedy with the same prompts and frozen retrieval as run_inference.py.
Resumable: finished (system, k, repeat, fold) blocks are skipped.
"""
import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import pandas as pd

from prompts import DEFAULT_VARIANT, INSTRUCTIONS, build_messages, load_fewshot
from run_inference import MODEL, HFGenerator, StubGenerator, load_adapted_model, load_base

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["system", "k", "repeat", "fold", "n_questions", "wall_s", "sum_latency_s", "prompt_tokens",
          "new_tokens", "energy_kwh", "gpu_kwh", "cpu_kwh", "ram_kwh", "gpu_nvml_j", "gpu_temp_c",
          "gpu", "torch", "codecarbon_ok"]
ORDER = {1: ["S1", "S2", "S3"], 2: ["S2", "S3", "S1"], 3: ["S3", "S1", "S2"]}


# ------------------------------------------------------------------ measuring helpers
class Nvml:
    """GPU board energy counter in joules (Volta and newer); silently unavailable elsewhere."""

    def __init__(self):
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:                                         # noqa: BLE001
            self.handle = None

    def joules(self):
        if self.handle is None:
            return None
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def temp(self):
        if self.handle is None:
            return None
        return self.nv.nvmlDeviceGetTemperature(self.handle, self.nv.NVML_TEMPERATURE_GPU)


def start_tracker(label):
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(project_name=label, save_to_file=False, log_level="error",
                                   measure_power_secs=2)
        tracker.start()
        return tracker
    except Exception as exc:                                      # noqa: BLE001
        print(f"CodeCarbon unavailable ({exc}); only latency and NVML energy are recorded")
        return None


def stop_tracker(tracker):
    out = dict(energy_kwh=None, gpu_kwh=None, cpu_kwh=None, ram_kwh=None, codecarbon_ok=False)
    if tracker is None:
        return out
    try:
        tracker.stop()
        data = tracker.final_emissions_data
        out.update(energy_kwh=data.energy_consumed, gpu_kwh=data.gpu_energy, cpu_kwh=data.cpu_energy,
                   ram_kwh=data.ram_energy, codecarbon_ok=True)
    except Exception as exc:                                      # noqa: BLE001
        print(f"CodeCarbon stop failed: {exc}")
    return out


# ------------------------------------------------------------------ summary
def summarize_csv(path, out_path):
    df = pd.read_csv(path)
    df["k"] = df["k"].astype(int)
    rows = []
    for (system, k), part in df.groupby(["system", "k"]):
        per_rep = []
        for repeat, rep in part.groupby("repeat"):
            n = rep.n_questions.sum()
            entry = {"repeat": repeat, "n": n, "latency_ms": 1000 * rep.sum_latency_s.sum() / n,
                     "new_tokens": rep.new_tokens.sum() / n, "prompt_tokens": rep.prompt_tokens.sum() / n}
            for col, key in (("energy_kwh", "wh_total"), ("gpu_kwh", "wh_gpu_codecarbon")):
                entry[key] = 1000 * rep[col].sum() / n if rep[col].notna().all() else float("nan")
            entry["wh_gpu_nvml"] = (rep.gpu_nvml_j.sum() / 3600 / n if rep.gpu_nvml_j.notna().all()
                                    else float("nan"))
            per_rep.append(entry)
        frame = pd.DataFrame(per_rep)
        row = {"system": system, "k": k, "repeats": len(frame), "questions_per_repeat": int(frame.n.iloc[0])}
        for col in ("wh_total", "wh_gpu_codecarbon", "wh_gpu_nvml", "latency_ms", "prompt_tokens", "new_tokens"):
            values = frame[col].dropna()
            row[col + "_mean"] = values.mean() if len(values) else float("nan")
            row[col + "_sd"] = statistics.stdev(values) if len(values) > 1 else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["k", "system"])
    summary.to_csv(out_path, index=False)
    show = summary[["system", "k", "repeats", "wh_total_mean", "wh_total_sd", "wh_gpu_nvml_mean",
                    "latency_ms_mean", "latency_ms_sd", "prompt_tokens_mean", "new_tokens_mean"]]
    print(show.round(4).to_string(index=False))
    return summary


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["S1", "S2", "S3"], choices=["S1", "S2", "S3"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3, help="untracked questions before each block")
    parser.add_argument("--per-fold", type=int, default=None, help="questions per fold (default: all 80)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(INSTRUCTIONS))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter-root", default=str(ROOT / "adapters"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default=None, help="raw per-block CSV (default results/energy/energy_runs_k{k}.csv)")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) if args.out else ROOT / "results" / "energy" / f"energy_runs_k{args.k}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out.with_name(out.stem.replace("runs", "summary") + ".csv")
    if args.summarize_only:
        summarize_csv(out, summary_path)
        return

    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    folds = json.loads((ROOT / "data" / "folds.json").read_text(encoding="utf-8"))
    frozen = {}
    for line in open(ROOT / "data" / "retrieval_top5.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        frozen[rec["row_id"]] = rec["retrieved_chunks"]
    done = set()
    if out.exists():
        done = {(r["system"], int(r["k"]), int(r["repeat"]), int(r["fold"])) for r in csv.DictReader(open(out))}
    new_file = not out.exists()

    nvml = Nvml()
    gpu_name, torch_version, shared = "dry-run", "-", None
    if not args.dry_run:
        import torch
        assert torch.cuda.is_available(), "no GPU found (use --dry-run for a CPU check)"
        gpu_name, torch_version = torch.cuda.get_device_name(0), torch.__version__
        print(f"GPU: {gpu_name}  torch {torch_version}  NVML energy counter: {'yes' if nvml.handle else 'no'}")

    with out.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for repeat in range(1, args.repeats + 1):
            order = [s for s in ORDER.get(repeat, ORDER[1]) if s in args.systems]
            for system in order:
                for fold in range(4):
                    if (system, args.k, repeat, fold) in done:
                        continue
                    rows = annotations[annotations.fold.astype(int) == fold].to_dict("records")[: args.per_fold]
                    examples = load_fewshot(fold) if system == "S2" else None
                    if args.dry_run:
                        generate = StubGenerator()
                    else:
                        if system == "S3":
                            model, tokenizer = load_adapted_model(
                                args.model, Path(args.adapter_root) / f"fold{fold}", fold, folds)
                        else:
                            shared = shared or load_base(args.model)
                            model, tokenizer = shared
                        generate = HFGenerator(model, tokenizer, args.max_new_tokens)

                    def one(row):
                        chunks = frozen[row["row_id"]][: args.k]
                        messages = build_messages(row["question"], chunks, examples=examples, variant=args.variant)
                        start = time.perf_counter()
                        _, prompt_tokens, new_tokens = generate(messages, row)
                        if not args.dry_run:
                            import torch
                            torch.cuda.synchronize()
                        return time.perf_counter() - start, prompt_tokens, new_tokens

                    for row in rows[: args.warmup]:          # warm-up: not measured
                        one(row)
                    tracker = start_tracker(f"{system}_k{args.k}_r{repeat}_f{fold}")
                    joules0, wall0 = nvml.joules(), time.perf_counter()
                    latency = prompt_total = new_total = 0.0
                    for row in rows:
                        seconds, prompt_tokens, new_tokens = one(row)
                        latency += seconds
                        prompt_total += prompt_tokens
                        new_total += new_tokens
                    wall = time.perf_counter() - wall0
                    joules1 = nvml.joules()
                    energy = stop_tracker(tracker)
                    writer.writerow({
                        "system": system, "k": args.k, "repeat": repeat, "fold": fold, "n_questions": len(rows),
                        "wall_s": round(wall, 2), "sum_latency_s": round(latency, 3),
                        "prompt_tokens": int(prompt_total), "new_tokens": int(new_total),
                        "gpu_nvml_j": None if joules0 is None or joules1 is None else round(joules1 - joules0, 1),
                        "gpu_temp_c": nvml.temp(), "gpu": gpu_name, "torch": torch_version, **energy})
                    stream.flush()
                    wh = energy["energy_kwh"]
                    print(f"{system} k={args.k} repeat {repeat} fold {fold}: {len(rows)} q, {wall:.0f}s, "
                          f"{'%.2f Wh' % (wh * 1000) if wh is not None else 'no CodeCarbon'}, "
                          f"{1000 * latency / len(rows):.0f} ms/query", flush=True)
                    if system == "S3" and not args.dry_run:
                        del model, generate
                        import gc
                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
    print()
    summarize_csv(out, summary_path)
    meta = {"args": vars(args), "python": platform.python_version(), "gpu": gpu_name, "torch": torch_version,
            "dtype": "float16", "greedy": True}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
