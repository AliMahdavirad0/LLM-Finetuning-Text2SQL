
import argparse
import random
import sqlite3
import time
import re
import os
import json

import sqlparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


SYSTEM_PROMPT = (
    "You are a text-to-SQL translator.\n"
    "Return exactly ONE SQLite query.\n"
    "Rules:\n"
    "- Use double quotes ONLY for table/column identifiers.\n"
    "- Use single quotes for string values.\n"
    "- Use ONLY tables/columns that exist in the given schema.\n"
    "Output only SQL, nothing else."
)

SQL_START_RE = re.compile(r"\b(select|with|insert|update|delete)\b", re.IGNORECASE)


def print_device_info(device: str):
    print("========== Eval Device Info ==========")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Requested device: {device}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"Using GPU: {torch.cuda.get_device_name(idx)} (index {idx})")
        print(f"VRAM total: {total_gb:.2f} GB")
    else:
        print("Using CPU")
    print("======================================\n")


def norm_sql_strict(s: str) -> str:
    s = s.strip().rstrip(";")
    s = sqlparse.format(
        s,
        keyword_case="lower",
        strip_comments=True,
        reindent=False
    )
    s = " ".join(s.split())
    return s


def norm_sql_noquotes(s: str) -> str:

    return norm_sql_strict(s.replace('"', ""))


def build_prompt(tokenizer, schema: str, question: str) -> str:
    user = f"Schema:\n{schema}\n\nQuestion:\n{question}\n\nSQL:"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user}\n"


def extract_sql(pred: str) -> str:
    pred = pred.strip()
    pred = pred.replace("```sql", "").replace("```", "").strip()
    pred = re.sub(r"^\s*assistant\s*", "", pred, flags=re.IGNORECASE)

    m = SQL_START_RE.search(pred)
    if m:
        pred = pred[m.start():].strip()


    if ";" in pred:
        pred = pred.split(";", 1)[0].strip() + ";"
    else:
        pred = pred.strip() + ";"

    return pred


def is_sql_valid(schema_sql: str, query_sql: str):

    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema_sql)
        con.execute("EXPLAIN QUERY PLAN " + query_sql.strip().rstrip(";"))
        con.close()
        return True, None
    except Exception as e:
        return False, str(e)


def _from_pretrained_dtype_compat(cls, name_or_dir: str, dtype, device_map=None):

    try:
        return cls.from_pretrained(name_or_dir, dtype=dtype, device_map=device_map)
    except TypeError:
        return cls.from_pretrained(name_or_dir, torch_dtype=dtype, device_map=device_map)


def load_tokenizer(tokenizer_source: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_dir: str, base_model: str | None, device: str, merge_adapter: bool):

    dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32


    is_adapter = (
        os.path.exists(os.path.join(model_dir, "adapter_config.json")) or
        os.path.exists(os.path.join(model_dir, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(model_dir, "adapter_model.bin"))
    )

    if is_adapter:
        if base_model is None:
            raise ValueError("This looks like a LoRA adapter folder. You must pass --base_model.")
        if PeftModel is None:
            raise RuntimeError("peft is not installed. Install it: pip install peft")

        base = _from_pretrained_dtype_compat(AutoModelForCausalLM, base_model, dtype=dtype, device_map=None)
        if device == "cuda" and torch.cuda.is_available():
            base = base.to("cuda")
        base.eval()

        model = PeftModel.from_pretrained(base, model_dir).eval()

        if merge_adapter:

            try:
                model = model.merge_and_unload()
                model.eval()
            except Exception:
                pass
        return model


    model = _from_pretrained_dtype_compat(AutoModelForCausalLM, model_dir, dtype=dtype, device_map=None)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    return model.eval()


@torch.inference_mode()
def generate_sql(model, tok, schema: str, question: str, max_new_tokens: int, device: str):
    prompt = build_prompt(tok, schema, question)
    inputs = tok(prompt, return_tensors="pt")

    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
    )


    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    pred = tok.decode(gen_ids, skip_special_tokens=True).strip()
    pred = extract_sql(pred)

    gen_token_count = int(gen_ids.numel())
    return pred, gen_token_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="merged model folder OR LoRA adapter folder")
    ap.add_argument("--base_model", default=None, help="required only if model_dir is a LoRA adapter")
    ap.add_argument("--eval_file", default="data/processed/wikisql_val.jsonl")
    ap.add_argument("--num_samples", type=int, default=200, help="0 => use all")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_examples", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--merge_adapter", action="store_true", help="if adapter, merge into base for faster eval")
    ap.add_argument("--save_jsonl", default="", help="optional: save per-sample predictions to a jsonl file")
    args = ap.parse_args()


    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    random.seed(args.seed)
    print_device_info(device)


    data = load_dataset("json", data_files={"eval": args.eval_file})["eval"]
    total = len(data)
    if args.num_samples == 0 or args.num_samples > total:
        idxs = list(range(total))
    else:
        idxs = random.sample(range(total), args.num_samples)


    tok_source = args.base_model if args.base_model else args.model_dir
    tok = load_tokenizer(tok_source)


    model = load_model(args.model_dir, args.base_model, device, args.merge_adapter)


    em_strict = 0
    em_noquotes = 0
    valid = 0

    shown = 0
    t0 = time.time()
    total_gen_tokens = 0


    save_f = None
    if args.save_jsonl:
        os.makedirs(os.path.dirname(args.save_jsonl) or ".", exist_ok=True)
        save_f = open(args.save_jsonl, "w", encoding="utf-8")

    for k, i in enumerate(idxs, start=1):
        ex = data[i]
        schema = ex["schema"]
        question = ex["question"]
        gold = ex["sql"]

        pred, gen_tokens = generate_sql(model, tok, schema, question, args.max_new_tokens, device)
        total_gen_tokens += gen_tokens

        gold_s = norm_sql_strict(gold)
        pred_s = norm_sql_strict(pred)
        gold_nq = norm_sql_noquotes(gold)
        pred_nq = norm_sql_noquotes(pred)

        is_em_strict = (pred_s == gold_s)
        is_em_nq = (pred_nq == gold_nq)
        is_valid, err = is_sql_valid(schema, pred)

        em_strict += int(is_em_strict)
        em_noquotes += int(is_em_nq)
        valid += int(is_valid)


        if save_f is not None:
            save_f.write(json.dumps({
                "question": question,
                "gold": gold,
                "pred": pred,
                "em_strict": is_em_strict,
                "em_noquotes": is_em_nq,
                "valid_sql": is_valid,
                "valid_error": err,
                "schema": schema,
            }, ensure_ascii=False) + "\n")

        if args.show_examples > 0 and (not is_em_nq) and shown < args.show_examples:
            shown += 1
            print("\n" + "=" * 90)
            print(f"[FAIL EXAMPLE #{shown}] sample {k}/{len(idxs)}")
            print("Question:", question)
            print("GOLD:", gold)
            print("PRED:", pred)
            print("EM(strict):", is_em_strict, " | EM(noQuotes):", is_em_nq)
            print("ValidSQL:", is_valid)
            if err:
                print("ValidSQL error:", err)
            print("=" * 90 + "\n")


        if k % args.log_every == 0 or k == len(idxs):
            elapsed = time.time() - t0
            per_item = elapsed / k
            eta = per_item * (len(idxs) - k)
            tok_s = (total_gen_tokens / elapsed) if elapsed > 0 else 0.0

            print(
                f"[{k:>4}/{len(idxs)}] "
                f"EM_strict={em_strict/k:.3f}  EM_noQuotes={em_noquotes/k:.3f}  Valid={valid/k:.3f}  "
                f"tok/s={tok_s:.1f}  avg={per_item:.2f}s  ETA={eta/60:.1f}m"
            )

    if save_f is not None:
        save_f.close()

    n = len(idxs)
    print("\n========== FINAL ==========")
    print(f"Samples:        {n}   (from total={total})")
    print(f"ExactMatch:     {em_strict/n:.3f}  ({em_strict}/{n})   [strict]")
    print(f"ExactMatch:     {em_noquotes/n:.3f}  ({em_noquotes}/{n})   [noQuotes]")
    print(f"ValidSQL:       {valid/n:.3f}  ({valid}/{n})")
    if args.save_jsonl:
        print(f"Saved preds to: {args.save_jsonl}")
    print("===========================")


if __name__ == "__main__":
    main()
