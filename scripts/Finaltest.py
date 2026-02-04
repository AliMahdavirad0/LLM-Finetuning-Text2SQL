
import argparse
import random
import sqlite3
import re
import time

import sqlparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = (
    "You are a text-to-SQL translator. "
    "Output ONLY one SQL query in SQLite dialect. "
    "Do not explain."
)

SQL_START_RE = re.compile(r"\b(select|with|insert|update|delete)\b", re.IGNORECASE)


def log_print(msg, fh=None):
    print(msg)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


def print_device_info(fh=None):
    log_print("========== Final Test Eval ==========", fh)
    log_print(f"PyTorch: {torch.__version__}", fh)
    log_print(f"CUDA available: {torch.cuda.is_available()}", fh)
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        log_print(f"Using GPU: {torch.cuda.get_device_name(idx)} (index {idx})", fh)
    else:
        log_print("Using CPU", fh)
    log_print("=====================================\n", fh)


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
    pred = pred.strip().replace("```sql", "").replace("```", "").strip()
    pred = re.sub(r"^\s*assistant\s*", "", pred, flags=re.IGNORECASE)

    m = SQL_START_RE.search(pred)
    if m:
        pred = pred[m.start():].strip()

    return (pred.split(";", 1)[0].strip() + ";")


def norm_sql(s: str) -> str:
    s = s.strip().rstrip(";")
    s = sqlparse.format(
        s, keyword_case="lower",
        identifier_case=None,
        strip_comments=True,
        reindent=False
    )
    return " ".join(s.split())


def is_sql_valid(schema_sql: str, query_sql: str) -> bool:
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema_sql)
        con.execute("EXPLAIN QUERY PLAN " + query_sql.strip().rstrip(";"))
        con.close()
        return True
    except Exception:
        return False


def load_model(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    return tok, model.eval()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--eval_file", default="data/processed/wikisql_test.jsonl")
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_log", default="logs/test_eval_result.txt")
    args = ap.parse_args()

    Path = __import__("pathlib").Path
    Path(args.save_log).parent.mkdir(parents=True, exist_ok=True)

    with open(args.save_log, "w", encoding="utf-8") as fh:

        random.seed(args.seed)
        print_device_info(fh)

        data = load_dataset("json", data_files={"test": args.eval_file})["test"]
        total = len(data)

        idxs = (
            random.sample(range(total), args.num_samples)
            if args.num_samples and args.num_samples < total
            else list(range(total))
        )

        n = len(idxs)
        log_print(f"Test file: {args.eval_file}", fh)
        log_print(f"Total test rows: {total}", fh)
        log_print(f"Samples used: {n}\n", fh)

        tok, model = load_model(args.model_dir)

        em = valid = 0
        t0 = time.time()

        for k, i in enumerate(idxs, start=1):
            ex = data[i]
            prompt = build_prompt(tok, ex["schema"], ex["question"])

            inputs = tok(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
            )

            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            pred = extract_sql(tok.decode(gen_ids, skip_special_tokens=True).strip())

            gold_n = norm_sql(ex["sql"])
            pred_n = norm_sql(pred)

            em += (pred_n == gold_n)
            valid += is_sql_valid(ex["schema"], pred)

            if k % args.log_every == 0 or k == n:
                elapsed = time.time() - t0
                log_print(
                    f"[{k}/{n}] EM={em/k:.3f}  Valid={valid/k:.3f}  avg_time={elapsed/k:.2f}s",
                    fh
                )

        em_acc = em / n
        valid_acc = valid / n

        log_print("\n========== FINAL TEST RESULT ==========", fh)
        log_print(f"Samples used: {n}", fh)
        log_print(f"Exact Match (EM): {em_acc:.3f}  ({em}/{n})", fh)
        log_print(f"Valid SQL:        {valid_acc:.3f}  ({valid}/{n})", fh)
        log_print("Saved log to: " + args.save_log, fh)
        log_print("=======================================", fh)


if __name__ == "__main__":
    main()
