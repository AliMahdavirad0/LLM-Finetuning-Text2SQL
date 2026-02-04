
import argparse
import json
import os
import random
import re
import sqlite3
from typing import Dict, List, Tuple

import sqlparse
import torch
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


def build_prompt(tokenizer, schema: str, question: str) -> str:
    user = f"Schema:\n{schema}\n\nQuestion:\n{question}\n\nSQL:"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user}\n"


def extract_sql(text: str) -> str:
    text = text.strip()
    text = text.replace("```sql", "").replace("```", "").strip()
    text = re.sub(r"^\s*assistant\s*", "", text, flags=re.IGNORECASE)

    m = SQL_START_RE.search(text)
    if m:
        text = text[m.start():].strip()

    if ";" in text:
        text = text.split(";", 1)[0].strip() + ";"
    else:
        text = text.strip() + ";"
    return text


def norm_sql_strict(s: str) -> str:
    s = s.strip().rstrip(";")
    s = sqlparse.format(
        s,
        keyword_case="lower",
        strip_comments=True,
        reindent=False,
    )
    return " ".join(s.split())


def norm_sql_noquotes(s: str) -> str:
    return norm_sql_strict(s.replace('"', ""))


def is_sql_valid(schema_sql: str, query_sql: str) -> bool:
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema_sql)
        con.execute("EXPLAIN QUERY PLAN " + query_sql.strip().rstrip(";"))
        con.close()
        return True
    except Exception:
        return False


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def print_device_info(device: str):
    print("========== Device Info ==========")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(idx)} (index {idx})")
        print(f"VRAM total: {total_gb:.2f} GB")
    print("=================================\n")


def load_tokenizer(source: str):
    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _from_pretrained_dtype_compat(name_or_dir: str, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(name_or_dir, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name_or_dir, torch_dtype=dtype)


def detect_adapter_dir(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "adapter_config.json"))
        or os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_model.bin"))
    )


def load_model_any(model_dir: str, base_model: str | None, device: str, merge_adapter: bool):

    dtype = torch.float16 if device == "cuda" else torch.float32

    if detect_adapter_dir(model_dir):
        if base_model is None:
            raise ValueError("model_dir looks like a LoRA adapter. Please pass --base_model.")
        if PeftModel is None:
            raise RuntimeError("peft not installed. Install: pip install peft")

        base = _from_pretrained_dtype_compat(base_model, dtype=dtype)
        if device == "cuda":
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

    model = _from_pretrained_dtype_compat(model_dir, dtype=dtype)
    if device == "cuda":
        model = model.to("cuda")
    return model.eval()


@torch.inference_mode()
def generate_sql(model, tok, schema: str, question: str, max_new_tokens: int, device: str) -> str:
    prompt = build_prompt(tok, schema, question)
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return extract_sql(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M-Instruct", help="base HF model id/dir")
    ap.add_argument("--finetuned_dir", required=True, help="LoRA adapter folder OR merged model folder")
    ap.add_argument("--eval_file", required=True, help="JSONL with schema/question/sql")
    ap.add_argument("--num_samples", type=int, default=200, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--merge_adapter", action="store_true")
    ap.add_argument("--show_improvements", type=int, default=5)
    ap.add_argument("--show_regressions", type=int, default=5)
    args = ap.parse_args()

    device = pick_device(args.device)
    random.seed(args.seed)

    print_device_info(device)

    data = read_jsonl(args.eval_file)
    if args.num_samples and args.num_samples > 0 and args.num_samples < len(data):
        data = random.sample(data, args.num_samples)

    n = len(data)
    print(f"Samples used: {n}\n")


    tok = load_tokenizer(args.base_model)

    base = load_model_any(args.base_model, base_model=None, device=device, merge_adapter=False)
    ft = load_model_any(args.finetuned_dir, base_model=args.base_model, device=device, merge_adapter=args.merge_adapter)

    base_em_strict = base_em_noq = base_valid = 0
    ft_em_strict = ft_em_noq = ft_valid = 0

    improvements: List[Tuple[Dict, str, str]] = []
    regressions: List[Tuple[Dict, str, str]] = []

    for ex in data:
        schema = ex["schema"]
        q = ex["question"]
        gold = ex["sql"]

        base_pred = generate_sql(base, tok, schema, q, args.max_new_tokens, device)
        ft_pred = generate_sql(ft, tok, schema, q, args.max_new_tokens, device)

        gold_s = norm_sql_strict(gold)
        base_s = norm_sql_strict(base_pred)
        ft_s = norm_sql_strict(ft_pred)

        gold_nq = norm_sql_noquotes(gold)
        base_nq = norm_sql_noquotes(base_pred)
        ft_nq = norm_sql_noquotes(ft_pred)

        b_em_s = (base_s == gold_s)
        f_em_s = (ft_s == gold_s)
        b_em_nq = (base_nq == gold_nq)
        f_em_nq = (ft_nq == gold_nq)

        b_valid = is_sql_valid(schema, base_pred)
        f_valid = is_sql_valid(schema, ft_pred)

        base_em_strict += int(b_em_s)
        ft_em_strict += int(f_em_s)
        base_em_noq += int(b_em_nq)
        ft_em_noq += int(f_em_nq)
        base_valid += int(b_valid)
        ft_valid += int(f_valid)


        if (not b_em_nq) and f_em_nq and len(improvements) < args.show_improvements:
            improvements.append((ex, base_pred, ft_pred))
        if b_em_nq and (not f_em_nq) and len(regressions) < args.show_regressions:
            regressions.append((ex, base_pred, ft_pred))

    def pct(x): return x / n if n else 0.0

    print("========== SUMMARY ==========")
    print(f"BASE  EM(strict):   {pct(base_em_strict):.3f} ({base_em_strict}/{n})")
    print(f"FT    EM(strict):   {pct(ft_em_strict):.3f} ({ft_em_strict}/{n})")
    print(f"BASE  EM(noQuotes): {pct(base_em_noq):.3f} ({base_em_noq}/{n})")
    print(f"FT    EM(noQuotes): {pct(ft_em_noq):.3f} ({ft_em_noq}/{n})")
    print(f"BASE  ValidSQL:     {pct(base_valid):.3f} ({base_valid}/{n})")
    print(f"FT    ValidSQL:     {pct(ft_valid):.3f} ({ft_valid}/{n})")
    print("----------------------------")
    print(f"DIFF EM(strict):   {(ft_em_strict - base_em_strict)/n:.3f}")
    print(f"DIFF EM(noQuotes): {(ft_em_noq - base_em_noq)/n:.3f}")
    print(f"DIFF ValidSQL:     {(ft_valid - base_valid)/n:.3f}")
    print("============================\n")

    if improvements:
        print("IMPROVEMENTS (FT fixed these):")
        for k, (ex, b, f) in enumerate(improvements, 1):
            print("-" * 90)
            print(f"[IMPROVE #{k}] Q: {ex['question']}")
            print("GOLD:", ex["sql"])
            print("BASE:", b)
            print("FT:  ", f)
        print()

    if regressions:
        print("REGRESSIONS (FT got worse here):")
        for k, (ex, b, f) in enumerate(regressions, 1):
            print("-" * 90)
            print(f"[REGRESS #{k}] Q: {ex['question']}")
            print("GOLD:", ex["sql"])
            print("BASE:", b)
            print("FT:  ", f)
        print()


if __name__ == "__main__":
    main()
