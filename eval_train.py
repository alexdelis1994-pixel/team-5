# ruff: noqa: T201
"""Прогон модели по русским сессиям train.json и расчёт метрик."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.models import LLMClient, load_llm, process_risk_detection


def is_english(session: dict) -> bool:
    text = " ".join(m["content"] for m in session["messages"])
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    cyrillic = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return latin > cyrillic


def format_dialogue(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def main() -> None:
    with Path("../train.json").open() as f:
        data = json.load(f)

    ru_sessions = [s for s in data if not is_english(s)]
    print(f"Русских сессий: {len(ru_sessions)}")

    llm: LLMClient = load_llm()

    tp = fp = fn = tn = 0
    errors: list[dict] = []

    for i, session in enumerate(ru_sessions):
        session_id = session["session_id"]
        expected_flags = session.get("expected_red_flags", [])
        expected_category = expected_flags[0]["category"] if expected_flags else None

        raw_text = format_dialogue(session["messages"])
        result = process_risk_detection(llm, raw_text)
        predicted_category = result["category"] if result else None

        expected_positive = expected_category is not None
        predicted_positive = predicted_category is not None

        if expected_positive and predicted_positive:
            if predicted_category == expected_category:
                tp += 1
                status = "TP"
            else:
                fp += 1
                fn += 1
                status = f"WRONG_CAT (pred={predicted_category})"
        elif expected_positive and not predicted_positive:
            fn += 1
            status = "FN"
        elif not expected_positive and predicted_positive:
            fp += 1
            status = f"FP (pred={predicted_category})"
        else:
            tn += 1
            status = "TN"

        print(f"[{i + 1:02d}] {session_id} | ожидалось={expected_category} | {status}")

        if status not in ("TP", "TN"):
            errors.append(
                {
                    "session_id": session_id,
                    "expected": expected_category,
                    "predicted": predicted_category,
                    "status": status,
                }
            )

        time.sleep(0.3)

    print("\n=== ИТОГО ===")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    if errors:
        print(f"\n=== ОШИБКИ ({len(errors)}) ===")
        for e in errors:
            print(f"  {e['session_id']}: ожидалось={e['expected']}, получено={e['predicted']} [{e['status']}]")


if __name__ == "__main__":
    main()
