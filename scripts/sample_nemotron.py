"""
从 mixed_sft.json 中过滤并采样 Nemotron 通用数据，
输出约 1000 条干净的通用 SFT 数据用于防止灾难性遗忘。

用法：
  python scripts/sample_nemotron.py \
      --input  data/mixed_sft.json \
      --output data/nemotron_sampled.json \
      --count  1000
"""

import argparse
import json
import random
from collections import Counter


def is_valid(sample: dict) -> bool:
    output = sample.get("output", "") or ""
    instruction = sample.get("instruction", "") or ""

    # 过滤 tool_call 类数据（不适合通用对话训练）
    if "<tool_call>" in output or "<tool_call>" in instruction:
        return False

    # 过滤超长 think 链（math 深度推理，占 token 太多）
    if "<think>" in output and len(output) > 3000:
        return False

    # 过滤空内容
    if not instruction.strip() or not output.strip():
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/mixed_sft.json")
    parser.add_argument("--output", default="data/nemotron_sampled.json")
    parser.add_argument("--count",  type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, encoding="utf-8") as f:
        all_data = json.load(f)

    # 只取 nemotron 数据
    nemotron = [d for d in all_data if d.get("source", "").startswith("nemotron")]
    print(f"Nemotron 原始数据: {len(nemotron)} 条")
    print("source 分布:", dict(Counter(d["source"] for d in nemotron)))

    # 过滤
    valid = [d for d in nemotron if is_valid(d)]
    print(f"\n过滤后: {len(valid)} 条（过滤掉 {len(nemotron) - len(valid)} 条）")

    # 优先保留 chat 和 instruction_following，补充其他类型
    priority   = [d for d in valid if d["source"] in ("nemotron_chat", "nemotron_instruction_following")]
    supplement = [d for d in valid if d["source"] not in ("nemotron_chat", "nemotron_instruction_following")]

    print(f"优先类(chat+instruct): {len(priority)} 条")
    print(f"补充类(math+science):  {len(supplement)} 条")

    # 采样：尽量用优先类填满，不足时用补充类补
    target = args.count
    if len(priority) >= target:
        sampled = random.sample(priority, target)
    else:
        sampled = priority + random.sample(supplement, min(target - len(priority), len(supplement)))
    random.shuffle(sampled)

    # source 字段统一为 nemotron_general（训练时不需要细分）
    for s in sampled:
        s["source"] = "nemotron_general"

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\n最终采样: {len(sampled)} 条 → {args.output}")
    print("\n── 示例（第1条）──")
    print(json.dumps(sampled[0], ensure_ascii=False, indent=2)[:400])


if __name__ == "__main__":
    main()
