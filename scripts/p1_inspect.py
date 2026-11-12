"""P1 步骤 1：原始 CSV 结构探查（只读，不写库）。"""
import json
import hashlib
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def find_csv(raw_dir: Path) -> Path:
    for p in raw_dir.rglob("*.csv"):
        return p
    raise FileNotFoundError("data/raw 下未找到 CSV")


def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("无法解码 CSV", b"", 0, 1, "utf-8/gb18030 均失败")


def main() -> None:
    csv = find_csv(RAW_DIR)
    df = read_csv_any(csv)
    out = {
        "file": str(csv),
        "size_mb": round(csv.stat().st_size / 1024 / 1024, 2),
        "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_per_column": {c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns},
        "app_counts": {str(k): int(v) for k, v in df["app_name"].value_counts().items()} if "app_name" in df else {},
        "package_counts": {str(k): int(v) for k, v in df["package_name"].value_counts().items()} if "package_name" in df else {},
        "score_counts": {str(k): int(v) for k, v in df["score"].value_counts().sort_index().items()} if "score" in df else {},
    }
    if "at_utc" in df:
        ts = pd.to_datetime(df["at_utc"], errors="coerce", utc=True)
        out["date_parse_fail"] = int(ts.isna().sum())
        out["date_min"] = str(ts.min())
        out["date_max"] = str(ts.max())
        out["per_app_date_range"] = {
            app: [str(ts[df["app_name"] == app].min()), str(ts[df["app_name"] == app].max())]
            for app in df["app_name"].unique()
        }
    if "content" in df:
        content = df["content"].astype(str)
        dup_id = int(df["review_id"].duplicated().sum()) if "review_id" in df else -1
        dup_content = int(content.duplicated().sum())
        ascii_ratio = float(content.str.replace(r"[^\x00-\x7F]", "", regex=True).str.len().sum() / max(1, content.str.len().sum()))
        out["duplicate_review_id"] = dup_id
        out["duplicate_content"] = dup_content
        out["content_len_median"] = float(content.str.len().median())
        out["content_len_max"] = int(content.str.len().max())
        out["content_short_lt5"] = int((content.str.len() < 5).sum())
        out["ascii_char_ratio"] = round(ascii_ratio, 4)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
