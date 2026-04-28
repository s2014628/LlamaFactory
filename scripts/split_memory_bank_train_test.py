"""
Split memory_bank.json into train/test and export SFT files.

Outputs:
  - raw train/test (original structure with sid + experience fields)
  - sft train/test (instruction/input/output)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


FORWARD_INSTRUCTIONS = [
    (
        "Given the following experience summary, what is the corresponding SID token? "
        "Reply with the SID token only, in the exact format <SID_L1_X><SID_L2_Y><SID_L3_Z>, no other text.\n"
        "{desc}"
    ),
    (
        "Based on the experience description below, identify its SID token. "
        "Output only the SID token in the format <SID_L1_X><SID_L2_Y><SID_L3_Z>.\n"
        "{desc}"
    ),
    (
        "What SID token corresponds to the following experience?\n"
        "{desc}\n"
        "Reply with the SID token only, format: <SID_L1_X><SID_L2_Y><SID_L3_Z>."
    ),
]

BACKWARD_INSTRUCTIONS = [
    "What experience or strategy does the SID token {sid} represent?",
    "Describe the experience corresponding to SID token {sid}.",
    "The SID token is {sid}. What problem-solving experience does it encode?",
]


def get_desc(sample: dict) -> str:
    summary = str(sample.get("experience_summary", "")).strip()
    if summary:
        return summary
    return str(sample.get("experience_text", "")).strip()


def is_valid(sample: dict) -> bool:
    sid_token = sample.get("sid", {}).get("tokens", "")
    return bool(sid_token and get_desc(sample))


def to_sft(samples: list[dict], rng: random.Random) -> list[dict]:
    rows = []
    for sample in samples:
        sid_token = sample["sid"]["tokens"]
        desc = get_desc(sample)

        fwd = rng.choice(FORWARD_INSTRUCTIONS).format(desc=desc)
        rows.append({"instruction": fwd, "input": "", "output": sid_token})

        bwd = rng.choice(BACKWARD_INSTRUCTIONS).format(sid=sid_token)
        rows.append({"instruction": bwd, "input": "", "output": desc})
    return rows


def stratified_split(valid: list[dict], test_ratio: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for sample in valid:
        grouped[sample.get("dataset", "unknown")].append(sample)

    train, test = [], []
    for _, group in grouped.items():
        rng.shuffle(group)
        n_test = max(1, int(round(len(group) * test_ratio))) if len(group) > 1 else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="/data2/jxk/RQ-KMeans/rqkmeans_8k_64_32_16/memory_bank.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/data2/jxk/LlamaFactory/data",
    )
    parser.add_argument("--train_raw_name", default="memory_bank_train.json")
    parser.add_argument("--test_raw_name", default="memory_bank_test.json")
    parser.add_argument("--train_sft_name", default="memory_bank_sft_train.json")
    parser.add_argument("--test_sft_name", default="memory_bank_sft_test.json")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    valid = [x for x in raw if is_valid(x)]
    train_raw, test_raw = stratified_split(valid, args.test_ratio, rng)
    train_sft = to_sft(train_raw, rng)
    test_sft = to_sft(test_raw, rng)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train_raw": out_dir / args.train_raw_name,
        "test_raw": out_dir / args.test_raw_name,
        "train_sft": out_dir / args.train_sft_name,
        "test_sft": out_dir / args.test_sft_name,
    }

    with open(paths["train_raw"], "w", encoding="utf-8") as f:
        json.dump(train_raw, f, ensure_ascii=False, indent=2)
    with open(paths["test_raw"], "w", encoding="utf-8") as f:
        json.dump(test_raw, f, ensure_ascii=False, indent=2)
    with open(paths["train_sft"], "w", encoding="utf-8") as f:
        json.dump(train_sft, f, ensure_ascii=False, indent=2)
    with open(paths["test_sft"], "w", encoding="utf-8") as f:
        json.dump(test_sft, f, ensure_ascii=False, indent=2)

    print("[split done]")
    print(f"  valid samples: {len(valid)}")
    print(f"  train raw: {len(train_raw)} -> {paths['train_raw']}")
    print(f"  test  raw: {len(test_raw)} -> {paths['test_raw']}")
    print(f"  train sft: {len(train_sft)} -> {paths['train_sft']}")
    print(f"  test  sft: {len(test_sft)} -> {paths['test_sft']}")


if __name__ == "__main__":
    main()
