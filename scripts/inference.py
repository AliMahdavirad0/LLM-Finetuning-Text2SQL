
import argparse
import re
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


def print_device_info(device: str):
    print("========== Inference Device Info ==========")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        idx = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(idx)} (index {idx})")
    print("===========================================\n")


def load_tokenizer(tok_source: str):
    tok = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _from_pretrained_dtype_compat(name_or_dir: str, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(name_or_dir, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name_or_dir, torch_dtype=dtype)


def load_model(model_dir: str, base_model: str | None, device: str, merge_adapter: bool):

    dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32


    import os
    is_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))

    if is_adapter:
        if base_model is None:
            raise ValueError("model_dir looks like a LoRA adapter. Please pass --base_model.")
        if PeftModel is None:
            raise RuntimeError("peft is not installed. Install: pip install peft")

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
def generate_sql(model, tok, schema: str, question: str, max_new_tokens: int, device: str):
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
    ap.add_argument("--model_dir", required=True, help="LoRA adapter folder OR merged model folder")
    ap.add_argument("--base_model", default=None, help="required only if model_dir is a LoRA adapter")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--merge_adapter", action="store_true", help="if adapter, merge for faster inference")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    print_device_info(device)


    tok_source = args.base_model if args.base_model else args.model_dir
    tok = load_tokenizer(tok_source)

    model = load_model(args.model_dir, args.base_model, device, args.merge_adapter)

    sql = generate_sql(model, tok, args.schema, args.question, args.max_new_tokens, device)
    print(sql)


if __name__ == "__main__":
    main()
