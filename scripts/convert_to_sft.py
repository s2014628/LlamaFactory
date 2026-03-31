"""
将 augment.json (JSONL格式) 转换为 LlamaFactory SFT 训练格式。

支持三种原始格式：
  A: 描述 → SID  ("The SID that may correspond to... Corresponding SID: <SID...>")
  B: SID → 描述  ("<SID...> The experience trajectory for this SID is:\n描述")
  C: SID → 描述  ("<SID...>\n描述" 纯换行，无桥接词)

输出格式（LlamaFactory alpaca）:
  {"instruction": "...", "input": "", "output": "..."}

用法:
  python scripts/convert_to_sft.py \
      --input data/augment.json \
      --output data/augment_sft.json
"""

import argparse
import json
import re
from pathlib import Path

SID_PAT = r"<SID_L1_\d+><SID_L2_\d+><SID_L3_\d+>"


def convert_entry(text: str):
    text = text.strip()

    # ── 格式 A：描述 → SID ──────────────────────────────────────────
    # "The SID that may correspond to the following experience trajectory is:\n{desc}\nCorresponding SID: <SID...>"
    m = re.match(
        r"The SID that may correspond to the following experience trajectory is:\n"
        r"(.*?)\n"
        r"Corresponding SID:\s*(" + SID_PAT + r")\s*$",
        text,
        re.DOTALL,
    )
    if m:
        description = m.group(1).strip()
        sid = m.group(2).strip()
        return {
            "instruction": (
                "What is the SID corresponding to the following experience trajectory? "
                "Reply with the SID token only, in the exact format <SID_L1_X><SID_L2_Y><SID_L3_Z>, no other text.\n"
                + description
            ),
            "input": "",
            "output": sid,
        }

    # ── 格式 B：SID → 描述（带桥接词）──────────────────────────────
    # "<SID...> The experience trajectory for this SID is:\n{desc}"
    m = re.match(
        r"(" + SID_PAT + r")\s+The experience trajectory for this SID is:\n(.*)",
        text,
        re.DOTALL,
    )
    if m:
        sid = m.group(1).strip()
        description = m.group(2).strip()
        return {
            "instruction": f"Describe the experience trajectory for SID {sid}.",
            "input": "",
            "output": description,
        }

    # ── 格式 C：SID → 描述（纯换行，无桥接词）──────────────────────
    # "<SID...>\n{desc}"
    m = re.match(
        r"(" + SID_PAT + r")\s*\n(.*)",
        text,
        re.DOTALL,
    )
    if m:
        sid = m.group(1).strip()
        description = m.group(2).strip()
        return {
            "instruction": f"Describe the experience trajectory for SID {sid}.",
            "input": "",
            "output": description,
        }

    return None  # 无法识别的格式


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/augment.json")
    parser.add_argument("--output", default="data/augment_sft.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    results = []
    skipped = []

    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[警告] 第{lineno}行 JSON 解析失败: {e}")
                continue

            text = item.get("text", "")
            converted = convert_entry(text)
            if converted:
                results.append(converted)
            else:
                skipped.append((lineno, text[:80]))

    # 统计方向分布
    sid2desc = sum(1 for r in results if r["instruction"].startswith("Describe"))
    desc2sid = sum(1 for r in results if r["instruction"].startswith("What is"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"转换完成: {len(results)} 条 → {output_path}")
    print(f"  SID → 描述: {sid2desc} 条")
    print(f"  描述 → SID: {desc2sid} 条")
    print(f"  跳过(无法识别): {len(skipped)} 条")

    if skipped:
        print("\n跳过的样本（前5条）:")
        for lineno, preview in skipped[:5]:
            print(f"  第{lineno}行: {preview!r}")

    print("\n示例输出（前2条）:")
    for r in results[:2]:
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
