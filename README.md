
##  Project Overview

This project builds a **Text-to-SQL system** by fine-tuning a language model to translate:

**Natural language questions ➜ SQL queries**

The pipeline covers:

- Dataset preprocessing  
- LoRA / QLoRA fine-tuning  
- Model evaluation  
- Inference tools  
- Web demo  

---

##  Example Task

**Schema**
```sql
CREATE TABLE users (
    id INTEGER,
    name TEXT,
    age INTEGER,
    city TEXT
);
````

**Question**

```
Show all users older than 30
```

**Model Output**

```sql
SELECT name FROM users WHERE age > 30;
```

---
**More Example Inputs**

| Question                           | Output SQL                                          |
| ---------------------------------- | --------------------------------------------------- |
| List all cities                    | `SELECT city FROM users;`                           |
| Show names of users in Paris       | `SELECT name FROM users WHERE city = 'Paris';`      |
| Count number of users              | `SELECT COUNT(*) FROM users;`                       |
| Find oldest user                   | `SELECT name FROM users ORDER BY age DESC LIMIT 1;` |
| Show all products with price > 100 | `SELECT * FROM products WHERE price > 100;`         |

##  Project Structure

```
Text2SQL-Finetune/
│
├── scripts/
│   ├── preprocess_wikisql.py        # Convert WikiSQL to training JSONL
│   ├── preprocess_wikisql_test.py   # Prepare test set
│   ├── train_lora.py                # Fine-tuning script
│   ├── eval_simple.py               # Evaluation metrics
│   ├── inference.py                 # Ask your own question
│   ├── demo_compare.py              # Base vs finetuned comparison
│   ├── Finaltest.py                 # Full test evaluation
│   └── app_gradio.py                # Web interface
│
├── data/
│   ├── raw_wikisql/                 # Original dataset files
│   └── processed/                   # Converted JSONL files
│
├── outputs/
│   └── smollm2_360m_wikisql_lora/    # Saved LoRA adapter
│
├── logs/
│   ├── loss_curve.png
│   ├── loss_log.csv
│   └── evaluation files
│
├── scripts.txt                      # One-line commands
└── README.md
```

---

##  Installation

```bash
pip install torch transformers datasets peft sqlparse matplotlib gradio
```

Optional (for low VRAM GPUs):

```bash
pip install bitsandbytes
```

---

##  Workflow

### 1️ Dataset Preparation

```bash
python scripts/preprocess_wikisql.py --out_dir data/processed
```

### 2️ Model Training

```bash
python scripts/train_lora.py --epochs 1 --batch_size 2 --grad_accum 16
```

### 3️ Evaluation

```bash
python scripts/eval_simple.py \
  --model_dir outputs/smollm2_360m_wikisql_lora \
  --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
  --eval_file data/processed/wikisql_val.jsonl \
  --merge_adapter
```

### 4️ Inference

```bash
python scripts/inference.py \
  --model_dir outputs/smollm2_360m_wikisql_lora \
  --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
  --schema 'CREATE TABLE products (id INT, price INT);' \
  --question "Show products cheaper than 50"
```

### 5️ Compare Base vs Finetuned

```bash
python scripts/demo_compare.py
```

### 6️ Run Web Demo

```bash
python scripts/app_gradio.py
```

Open:

```
http://127.0.0.1:7860
```

---

##  Training Logs

During training, logs are saved in:

```
logs/loss_log.csv
logs/loss_curve.png
```

### Example Loss Curve

![Training Curve](images/training_curve.png)

---



## Observations

* Model learns SQL structure after fine-tuning
* Outputs become more consistent
* Syntax errors reduce over time

---
