"""P1 校验脚本：检查 SQLite 表结构、条数、隐私字段是否入库、抽样可复现性。"""
import hashlib
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "feedbacklens.db"
SAMPLE = ROOT / "data" / "samples" / "sample_450.csv"

conn = sqlite3.connect(DB)
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print("tables:", tables)
print("feedback columns:", [r[1] for r in conn.execute("PRAGMA table_info(feedback)")])
print("feedback rows:", conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
print("per product:", conn.execute(
    "SELECT p.name, COUNT(*) FROM feedback f JOIN products p ON p.id=f.product_id GROUP BY p.name ORDER BY p.name").fetchall())
print("score dist:", conn.execute(
    "SELECT score, COUNT(*) FROM feedback GROUP BY score ORDER BY score").fetchall())
print("batch:", conn.execute("SELECT filename, row_count, accepted_count, rejected_count FROM import_batches").fetchall())
print("date range:", conn.execute("SELECT MIN(review_at), MAX(review_at) FROM feedback").fetchone())
print("privacy columns present:", [c for c in ("user_name", "user_image") if c in [r[1] for r in conn.execute("PRAGMA table_info(feedback)")]])
print("sample sha256:", hashlib.sha256(SAMPLE.read_bytes()).hexdigest())
print("sample lines:", sum(1 for _ in SAMPLE.open(encoding="utf-8-sig")) - 1)
conn.close()
