import argparse
import os
import time
import csv

import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


SYSTEM_PROMPT = (
    "You are a text-to-SQL translator. "
    "Output ONLY one SQL query in SQLite dialect. "
    "Do not explain."
)


def print_device_info():
    print("========== Device Info ==========")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"Using GPU: {name}")
        print(f"GPU index: {idx}")
        print(f"VRAM total: {total_gb:.2f} GB")
        try:
            alloc = torch.cuda.memory_allocated(idx) / (1024 ** 3)
            reserv = torch.cuda.memory_reserved(idx) / (1024 ** 3)
            print(f"VRAM allocated now: {alloc:.2f} GB")
            print(f"VRAM reserved now:  {reserv:.2f} GB")
        except Exception:
            pass
    else:
        print("Using CPU")
    print("=================================\n")


def build_prompt(tokenizer, schema: str, question: str) -> str:
    user = f"Schema:\n{schema}\n\nQuestion:\n{question}\n\nSQL:"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user}\n"


def tokenize_example(ex, tokenizer, max_len: int):
    prompt = build_prompt(tokenizer, ex["schema"], ex["question"])
    completion = ex["sql"].strip()
    eos = tokenizer.eos_token or ""

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion + eos, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + completion_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class PadCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tok.pad_token_id
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_n)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_n)
            batch["labels"].append(f["labels"] + [-100] * pad_n)
        return {k: torch.tensor(v) for k, v in batch.items()}


def pick_lora_targets(model, scope: str):
    mlp = ["up_proj", "down_proj", "gate_proj"]
    attn = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if scope == "mlp":
        wanted = mlp
    elif scope == "attn":
        wanted = attn
    else:
        wanted = attn + mlp

    names = set()
    for n, _ in model.named_modules():
        for w in wanted:
            if n.endswith(w):
                names.add(w)

    found = sorted(list(names))
    return found if found else mlp


class ConsoleLogCallback(TrainerCallback):

    def __init__(self):
        self._last_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time = time.time()
        print("Training started...\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        now = time.time()
        step_time = None
        if self._last_time is not None:
            step_time = now - self._last_time
        self._last_time = now

        pieces = []
        if "loss" in logs:
            pieces.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            pieces.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            pieces.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            pieces.append(f"epoch={logs['epoch']:.2f}")
        pieces.append(f"step={state.global_step}")
        if step_time is not None:
            pieces.append(f"dt={step_time:.2f}s")

        if torch.cuda.is_available() and state.global_step % 200 == 0:
            idx = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(idx) / (1024 ** 3)
            reserv = torch.cuda.memory_reserved(idx) / (1024 ** 3)
            pieces.append(f"vram_alloc={alloc:.2f}GB vram_res={reserv:.2f}GB")

        print("[LOG] " + " | ".join(pieces))


class LossPlotCallback(TrainerCallback):

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.train_steps, self.train_losses = [], []
        self.eval_steps, self.eval_losses = [], []
        self.csv_path = os.path.join(out_dir, "loss_log.csv")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["step", "train_loss", "eval_loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)

        if train_loss is not None:
            self.train_steps.append(step)
            self.train_losses.append(float(train_loss))

        if eval_loss is not None:
            self.eval_steps.append(step)
            self.eval_losses.append(float(eval_loss))

        if train_loss is not None or eval_loss is not None:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([step, train_loss if train_loss is not None else "", eval_loss if eval_loss is not None else ""])

    def on_train_end(self, args, state, control, **kwargs):
        if len(self.train_losses) == 0 and len(self.eval_losses) == 0:
            print("[WARN] No losses captured to plot.")
            return

        plt.figure()
        if len(self.train_losses) > 0:
            plt.plot(self.train_steps, self.train_losses, label="train_loss")
        if len(self.eval_losses) > 0:
            plt.plot(self.eval_steps, self.eval_losses, label="eval_loss")

        plt.xlabel("global_step")
        plt.ylabel("loss")
        plt.title("Training & Evaluation Loss")
        plt.legend()
        plt.grid(True)

        out_png = os.path.join(self.out_dir, "loss_curve.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[OK] Saved loss plot: {out_png}")
        print(f"[OK] Saved loss csv:  {self.csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--train_file", default="data/processed/wikisql_train.jsonl")
    ap.add_argument("--val_file", default="data/processed/wikisql_val.jsonl")
    ap.add_argument("--output_dir", default="outputs/smollm2_360m_wikisql_lora")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--lora_scope", choices=["mlp", "attn", "all"], default="mlp")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging (requires tensorboard installed)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print_device_info()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    use_qlora = False
    if args.load_in_4bit and torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            use_qlora = True
        except Exception:
            print("[WARN] bitsandbytes not found. Falling back to normal (non-4bit) LoRA.")

    print(f"QLoRA 4-bit enabled: {use_qlora}\n")

    if use_qlora:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=qconfig,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.to("cuda")

    target_modules = pick_lora_targets(model, args.lora_scope)
    print("LoRA target_modules:", target_modules)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        reserv = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        print(f"After model+LoRA load: vram_alloc={alloc:.2f}GB vram_res={reserv:.2f}GB\n")

    data = load_dataset("json", data_files={"train": args.train_file, "val": args.val_file})

    def map_fn(ex):
        return tokenize_example(ex, tokenizer, args.max_len)

    train_ds = data["train"].map(map_fn, remove_columns=data["train"].column_names)
    val_ds = data["val"].map(map_fn, remove_columns=data["val"].column_names)


    report_to = ["tensorboard"] if args.use_tensorboard else "none"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,

        logging_dir=args.log_dir,
        logging_strategy="steps",
        logging_steps=25,
        log_level="info",

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,

        fp16=torch.cuda.is_available(),
        report_to=report_to,
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer),
        callbacks=[
            ConsoleLogCallback(),
            LossPlotCallback(args.log_dir),
        ],
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saved to:", args.output_dir)
    print(f"Logs dir: {args.log_dir}")
    print("Loss plot: logs/loss_curve.png")
    print("Loss csv:  logs/loss_log.csv")
    if args.use_tensorboard:
        print("TensorBoard: tensorboard --logdir logs  (then open http://localhost:6006)")


if __name__ == "__main__":
    main()
