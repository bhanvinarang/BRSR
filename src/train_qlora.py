"""QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on one fold's training companies (S3).

    python src/train_qlora.py --fold 0
    python src/train_qlora.py --fold 0 --max-steps 3        # smoke test: checks memory and the loss

Settings fixed in advance (paper, Section IV-C): 4-bit NF4 base, LoRA r=16 / alpha=32 / dropout 0.05 on all
attention and MLP projections, learning rate 2e-4, 3 epochs, effective batch 16 (1 x 16 accumulation), fp16,
seed 42.  Also AdamW (no weight decay), cosine decay with 3% warmup, gradient clipping 1.0.  The loss is
computed on the answer tokens only.  The adapter is saved with train_meta.json, which run_inference.py checks
so that a test company can never be answered by an adapter that trained on it.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
END_OF_TURN = "<|im_end|>"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_examples(path, fold, folds):
    examples = [json.loads(line) for line in open(path, encoding="utf-8")]
    test_companies = {c for c, f in folds.items() if int(f) == int(fold)}
    train_companies = {e["company_id"] for e in examples}
    assert test_companies.isdisjoint(train_companies), f"test companies in training data: {test_companies & train_companies}"
    assert all(int(e["fold"]) == int(fold) for e in examples), "examples belong to a different fold"
    return examples, sorted(train_companies), sorted(test_companies)


def encode(example, tokenizer, max_len):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(example["target"] + END_OF_TURN, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None
    return prompt_ids, target_ids


def answer_loss(model, torch, prompt_ids, target_ids):
    """Cross-entropy over the answer tokens only; asks the model for just those logits to save memory."""
    ids = torch.tensor([prompt_ids + target_ids], device=model.device)
    n = len(target_ids)
    logits = None
    for keyword in ("logits_to_keep", "num_logits_to_keep"):
        try:
            logits = model(input_ids=ids, **{keyword: n + 1}).logits
            break
        except TypeError:
            continue
    if logits is None:
        logits = model(input_ids=ids).logits
    logits = logits[:, -(n + 1):-1, :].float()
    labels = torch.tensor([target_ids], device=model.device)
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke test)")
    parser.add_argument("--no-energy", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    assert torch.cuda.is_available(), "training needs a GPU (Colab T4 is enough)"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    folds = json.loads((ROOT / "data" / "folds.json").read_text(encoding="utf-8"))
    data_path = Path(args.data) if args.data else ROOT / "data" / "ft" / f"fold{args.fold}_train.jsonl"
    out_dir = Path(args.out) if args.out else ROOT / "adapters" / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples, train_companies, test_companies = read_examples(data_path, args.fold, folds)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = [(e, encode(e, tokenizer, args.max_len)) for e in examples]
    skipped = [e["row_id"] for e, enc in encoded if enc is None]
    data = [enc for _, enc in encoded if enc is not None]
    lengths = [len(p) + len(t) for p, t in data]
    print(f"{len(data)} examples ({len(skipped)} longer than {args.max_len} tokens skipped); "
          f"tokens mean {np.mean(lengths):.0f}, max {max(lengths)}")

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.float16,
                                                     device_map={"": 0})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                     torch_dtype=torch.float16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_MODULES,
                                              task_type="CAUSAL_LM", bias="none"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(data) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, round(0.03 * total_steps))
    schedule = lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda")

    tracker = None
    if not args.no_energy:
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(project_name=f"train_fold{args.fold}", output_dir=str(ROOT / "results"),
                                       output_file=f"energy_train_fold{args.fold}.csv", log_level="error")
            tracker.start()
        except Exception as exc:                      # noqa: BLE001 - energy logging is best effort
            print(f"energy tracking unavailable: {exc}")

    model.train()
    losses, step, start = [], 0, time.time()
    stop = False
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses = []
        for i, index in enumerate(order, 1):
            prompt_ids, target_ids = data[index]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = answer_loss(model, torch, prompt_ids, target_ids)
            scaler.scale(loss / args.grad_accum).backward()
            epoch_losses.append(loss.item())
            if i % args.grad_accum == 0 or i == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                recent = float(np.mean(epoch_losses[-args.grad_accum:]))
                losses.append({"epoch": epoch + 1, "step": step, "loss": recent})
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} loss {recent:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} elapsed {(time.time() - start) / 60:.1f} min", flush=True)
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
        print(f"epoch {epoch + 1} mean loss {np.mean(epoch_losses):.4f}")
        if stop:
            break

    energy_kwh = None
    if tracker is not None:
        try:
            tracker.stop()
            energy_kwh = tracker.final_emissions_data.energy_consumed
        except Exception as exc:                      # noqa: BLE001
            print(f"energy tracking failed: {exc}")
    model.save_pretrained(out_dir)
    meta = {
        "fold": args.fold, "train_companies": train_companies, "test_companies": test_companies,
        "n_examples": len(data), "skipped": skipped, "base_model": args.model, "args": vars(args),
        "losses": losses, "training_minutes": round((time.time() - start) / 60, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0), "energy_kwh": energy_kwh, "torch": torch.__version__,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved adapter to {out_dir} (peak GPU memory {meta['peak_gpu_gb']} GB, {meta['training_minutes']} min)")


if __name__ == "__main__":
    main()
