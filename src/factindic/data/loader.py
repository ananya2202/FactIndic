"""JSONL and schema-validated data loading and persistence."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads records from a JSON Lines (.jsonl) file."""
    records = []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error on line {idx + 1} of {path}: {e}")
    return records


def save_jsonl(
    records: List[Dict[str, Any]],
    path: str,
    append: bool = False
) -> None:
    """Saves records to a JSON Lines (.jsonl) file, creating parent directories if needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_schema(record: Dict[str, Any], required_keys: List[str]) -> bool:
    """Verifies that all required fields exist and are non-empty in the record."""
    for key in required_keys:
        if key not in record or record[key] is None:
            return False
    return True
