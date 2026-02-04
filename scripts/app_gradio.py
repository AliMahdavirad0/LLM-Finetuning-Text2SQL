
import argparse
import os
import re
import torch
import gradio as gr
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
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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


def _from_pretrained_dtype_compat(name_or_dir: str, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(name_or_dir, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name_or_dir, torch_dtype=dtype)


def load_tokenizer(source: str):
    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def detect_adapter_dir(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "adapter_config.json")) or
        os.path.exists(os.path.join(path, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(path, "adapter_model.bin"))
    )


def load_model(model_dir: str, base_model: str | None, device: str, merge_adapter: bool):
    dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

    if detect_adapter_dir(model_dir):
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


def make_app(model_dir: str, base_model: str | None, device: str, merge_adapter: bool, max_new_tokens: int):
    tok_source = base_model if (base_model is not None) else model_dir
    tok = load_tokenizer(tok_source)
    model = load_model(model_dir, base_model, device, merge_adapter)

    def device_banner():
        if device == "cuda" and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return f"GPU: {torch.cuda.get_device_name(idx)}"
        return "CPU mode"

    @torch.inference_mode()
    def infer(schema, question):
        schema = (schema or "").strip()
        question = (question or "").strip()
        if not schema or not question:
            return "Please fill both Schema and Question."

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
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        return extract_sql(text)

    with gr.Blocks() as demo:
        gr.Markdown("# Text → SQL (Local)")
        gr.Markdown(device_banner())

        with gr.Row():
            schema_box = gr.Textbox(lines=10, label="Schema (CREATE TABLE ...)", placeholder='e.g.\nCREATE TABLE "users" ("id" INTEGER, "name" TEXT);')
        with gr.Row():
            question_box = gr.Textbox(lines=2, label="Question", placeholder="e.g. List all users older than 30")

        with gr.Row():
            btn = gr.Button("Generate SQL")
            clear = gr.Button("Clear")

        out_box = gr.Textbox(lines=6, label="SQL Output")

        btn.click(infer, inputs=[schema_box, question_box], outputs=out_box)
        clear.click(lambda: ("", "", ""), outputs=[schema_box, question_box, out_box])

        gr.Markdown(
            "Note: The model may occasionally generate incorrect or incomplete SQL queries. "
            "Please review the output before executing it."
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="LoRA adapter folder OR merged model folder")
    ap.add_argument("--base_model", default=None, help="required only if model_dir is a LoRA adapter")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--merge_adapter", action="store_true", help="if adapter, merge for faster inference")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    app = make_app(
        model_dir=args.model_dir,
        base_model=args.base_model,
        device=device,
        merge_adapter=args.merge_adapter,
        max_new_tokens=args.max_new_tokens
    )
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
