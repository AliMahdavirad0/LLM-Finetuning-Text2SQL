
import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

DATA_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def sql_literal(v: str) -> str:
    v = str(v).strip()

    try:
        float(v)
        return v
    except ValueError:
        pass
    v = v.replace("'", "''")
    return f"'{v}'"


def build_schema(table_name: str, headers: list[str]) -> str:
    cols = ",\n  ".join([f"\"{h.replace('\"', '\"\"')}\" TEXT" for h in headers])
    return f'CREATE TABLE "{table_name}" (\n  {cols}\n);'


def build_sql(table_name: str, headers: list[str], sql_obj: dict) -> str:
    sel = sql_obj["sel"]
    agg = sql_obj["agg"]
    conds = sql_obj["conds"]

    sel_col = f"\"{headers[sel].replace('\"', '\"\"')}\""
    select_expr = f"{AGG_OPS[agg]}({sel_col})" if agg != 0 else sel_col

    where_parts = []
    for c in conds:
        col_i = c["column_index"]
        op_i = c["operator_index"]
        op = COND_OPS[op_i] if op_i < len(COND_OPS) else "="
        if op == "OP":
            op = "="
        col_name = f"\"{headers[col_i].replace('\"', '\"\"')}\""
        where_parts.append(f"{col_name} {op} {sql_literal(c['condition'])}")

    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    return f'SELECT {select_expr} FROM "{table_name}"{where_clause};'


def download_if_needed(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved: {out_path}")


def extract_if_needed(tar_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)

    for p in extract_dir.rglob("test.jsonl"):
        return
    print(f"Extracting: {tar_path} -> {extract_dir}")
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(path=extract_dir)
    print("Extract done.")


def find_data_dir(extract_dir: Path) -> Path:
    for p in extract_dir.rglob("test.jsonl"):
        return p.parent
    raise FileNotFoundError("Could not find test.jsonl after extraction.")


def load_tables(tables_path: Path) -> dict:
    id_to_table = {}
    with tables_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            id_to_table[t["id"]] = t
    return id_to_table


def iter_examples(main_path: Path, tables_path: Path):
    id_to_table = load_tables(tables_path)
    with main_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            table = id_to_table[row["table_id"]]
            row["table"] = table
            row.pop("table_id", None)


            conds = row["sql"]["conds"]
            row["sql"]["conds"] = [
                {"column_index": c[0], "operator_index": c[1], "condition": str(c[2])}
                for c in conds
            ]
            yield row


def write_split(rows_iter, out_path: Path, limit: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_iter:
            table = r["table"]
            table_name = (table.get("name") or table.get("id") or "table").strip() or (
                table.get("id") or "table"
            )
            headers = table["header"]

            schema = build_schema(table_name, headers)
            gold_sql = build_sql(table_name, headers, r["sql"])

            rec = {
                "schema": schema,
                "question": r["question"],
                "sql": gold_sql,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n += 1
            if limit and n >= limit:
                break
    print(f"Wrote {n} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--cache_dir", default="data/raw_wikisql")
    ap.add_argument("--limit_test", type=int, default=0, help="0 = all test rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    tar_path = cache_dir / "data.tar.bz2"
    extract_dir = cache_dir / "extracted"

    download_if_needed(DATA_URL, tar_path)
    extract_if_needed(tar_path, extract_dir)
    data_dir = find_data_dir(extract_dir)

    test_main = data_dir / "test.jsonl"
    test_tables = data_dir / "test.tables.jsonl"

    write_split(
        iter_examples(test_main, test_tables),
        out_dir / "wikisql_test.jsonl",
        args.limit_test,
    )

    print("Done.")
    print(f"- {out_dir / 'wikisql_test.jsonl'}")


if __name__ == "__main__":
    main()
