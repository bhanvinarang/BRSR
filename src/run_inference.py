"""Run S1 / S2 / S3 on the 320 questions and log every prediction to JSONL.

    python src/run_inference.py --system S1
    python src/run_inference.py --system S2
    python src/run_inference.py --system S3 --adapter-root adapters      # needs adapters/fold{0..3}
    python src/run_inference.py --system S1 --folds 0 --limit 8           # quick look
    python src/run_inference.py --system S1 --dry-run                     # no GPU, stub model

All systems read the same frozen retrieval (data/retrieval_top5.jsonl), use greedy decoding and the
same fp16 base model; S3 merges that fold's LoRA adapter into it, so test companies are only ever
answered by an adapter that did not train on them.  Each JSONL line holds:
    system, fold, question_id, retrieved_chunks, raw_output, parsed_answer, latency_s, prompt_tokens, new_tokens
Re-running appends only the questions that are not yet in the file.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd

from metrics import evaluate_jsonl, summarize
from parse_output import parse_model_output
from prompts import DEFAULT_VARIANT, INSTRUCTIONS, build_messages, load_fewshot

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ------------------------------------------------------------------ generators
class StubGenerator:
    """CPU stand-in used by --dry-run: cycles through realistic reply styles, including bad ones."""

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, row):
        from prompts import gold_to_target, target_json
        self.calls += 1
        mode = self.calls % 5
        target = gold_to_target(row)
        if mode == 1:
            return target_json(target), 0, 0
        if mode == 2:
            return "```json\n" + target_json(target) + "\n```", 0, 0
        if mode == 3:
            return '{"answer": "not disclosed"}', 0, 0
        if mode == 4:
            return '{"answer": "12,345", "unit": "GJ"}', 0, 0
        return "I think the answer is in the table.", 0, 0


class HFGenerator:
    def __init__(self, model, tokenizer, max_new_tokens):
        import torch
        self.torch, self.model, self.tokenizer, self.max_new_tokens = torch, model, tokenizer, max_new_tokens

    def __call__(self, messages, row=None):
        torch = self.torch
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens, temperature=None, top_p=None,
                top_k=None, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True), int(inputs["input_ids"].shape[1]), int(len(new_ids))


def load_base(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map={"": 0})
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": 0})
    model.eval()
    return model, tokenizer


def check_adapter_meta(meta, fold, folds):
    """The adapter must come from this fold's training split and never have seen the test companies."""
    test_companies = {c for c, f in folds.items() if int(f) == int(fold)}
    assert int(meta["fold"]) == int(fold), f"adapter was trained for fold {meta['fold']}, not {fold}"
    leaked = test_companies & set(meta["train_companies"])
    assert not leaked, f"adapter for fold {fold} was trained on test companies: {sorted(leaked)}"


def load_adapted_model(model_name, adapter_dir, fold, folds):
    from peft import PeftModel
    meta = json.loads((Path(adapter_dir) / "train_meta.json").read_text(encoding="utf-8"))
    check_adapter_meta(meta, fold, folds)
    model, tokenizer = load_base(model_name)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.eval()
    return model, tokenizer


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=["S1", "S2", "S3"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter-root", default=str(ROOT / "adapters"))
    parser.add_argument("--k", type=int, default=5, help="number of retrieved chunks shown (1-5)")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(INSTRUCTIONS))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="questions per fold (quick checks)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true", help="stub model, no GPU")
    args = parser.parse_args()

    out = Path(args.out) if args.out else ROOT / "results" / f"predictions_{args.system}_k{args.k}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    folds = json.loads((ROOT / "data" / "folds.json").read_text(encoding="utf-8"))
    frozen = {}
    for line in open(ROOT / "data" / "retrieval_top5.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        frozen[rec["row_id"]] = rec["retrieved_chunks"]
    done = set()
    if out.exists():
        done = {(r["system"], r["question_id"]) for r in map(json.loads, open(out, encoding="utf-8"))}

    meta = {"args": vars(args), "python": platform.python_version(), "greedy": True, "dtype": "float16"}
    shared = None
    for fold in args.folds:
        rows = annotations[annotations.fold.astype(int) == fold].to_dict("records")[: args.limit]
        examples = load_fewshot(fold) if args.system == "S2" else None
        if args.dry_run:
            generate = StubGenerator()
        else:
            import torch
            assert torch.cuda.is_available(), "no GPU found (use --dry-run for a CPU check)"
            if args.system == "S3":
                model, tokenizer = load_adapted_model(args.model, Path(args.adapter_root) / f"fold{fold}", fold, folds)
            else:
                shared = shared or load_base(args.model)
                model, tokenizer = shared
            generate = HFGenerator(model, tokenizer, args.max_new_tokens)
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["torch"] = torch.__version__
        with out.open("a", encoding="utf-8") as stream:
            for row in rows:
                if (args.system, row["row_id"]) in done:
                    continue
                chunks = frozen[row["row_id"]][: args.k]
                messages = build_messages(row["question"], chunks, examples=examples, variant=args.variant)
                start = time.perf_counter()
                raw, prompt_tokens, new_tokens = generate(messages, row)
                if not args.dry_run:
                    import torch
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                stream.write(json.dumps({
                    "system": args.system, "fold": fold, "question_id": row["row_id"],
                    "retrieved_chunks": chunks, "raw_output": raw,
                    "parsed_answer": parse_model_output(raw), "latency_s": round(latency, 4),
                    "prompt_tokens": prompt_tokens, "new_tokens": new_tokens,
                }, ensure_ascii=False) + "\n")
                stream.flush()
        print(f"fold {fold}: {len(rows)} questions processed", flush=True)
        if args.system == "S3" and not args.dry_run:
            del model, generate
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    scored = evaluate_jsonl(out, annotations, system=args.system)
    s = summarize(scored)
    abstain = sum(r["pred_kind"] == "not_disclosed" for r in scored) / max(len(scored), 1)
    print(f"\n{args.system}  n={s['n_scored']}  accuracy {s['accuracy']:.3f}  FAR {s['far']:.3f}  UAR {s['uar']:.3f}  "
          f"format failures {s['format_failure_rate']:.3f}  parse errors {s['parse_error_rate']:.3f}  "
          f"says 'not disclosed' {abstain:.3f}")


if __name__ == "__main__":
    main()
