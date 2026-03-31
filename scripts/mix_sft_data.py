"""
将领域 SFT 数据（augment_sft.json）与通用 SFT 数据混合。

用法：
  # 第一步：探查 Nemotron 数据集字段（不下载全量数据）
  python scripts/mix_sft_data.py --inspect

  # 第二步：正式混合（领域:通用 = 1:2）
  python scripts/mix_sft_data.py \
      --domain_file data/augment_sft.json \
      --general_count 9000 \
      --output data/mixed_sft.json

依赖：
  pip install datasets
"""

import argparse
import json
import random

NEMOTRON_DATASET = "nvidia/Nemotron-Cascade-2-SFT-Data"
# 可选子集: math, science, chat, instruction_following, safety, conversational_agent, swe, terminal_agent
NEMOTRON_CONFIGS = ["chat", "instruction_following", "math", "science"]


# ──────────────────────────────────────────────────────────────────────────────
# 探查 Nemotron 数据集字段，帮助确认字段名
# ──────────────────────────────────────────────────────────────────────────────
def inspect_nemotron():
    from datasets import load_dataset

    for config in NEMOTRON_CONFIGS:
        print(f"\n{'='*60}")
        print(f"[探查] 子集: {config}")
        try:
            ds = load_dataset(NEMOTRON_DATASET, config, split="train", streaming=True)
            for i, sample in enumerate(ds.take(2)):
                print(f"\n  ── 样本 {i} ──")
                for k, v in sample.items():
                    val_preview = str(v)[:200] if v else "(空)"
                    print(f"    {k!r}: {val_preview}")
        except Exception as e:
            print(f"  [错误] {e}")
    print("\n请根据上面的字段名确认 convert_nemotron() 函数是否需要修改。")


# ──────────────────────────────────────────────────────────────────────────────
# 将 Nemotron 单条样本转换为 instruction/input/output 格式
# !! 根据 inspect() 的结果修改这里 !!
# ──────────────────────────────────────────────────────────────────────────────
def convert_nemotron(example: dict) -> dict | None:
    """
    Nemotron 数据集字段格式（已通过 --inspect 确认）：
      messages: [
        {"role": "system",    "content": "..."},  # 可选
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
      ]
    """
    # ── 主格式：messages 列表 ──
    if "messages" in example:
        msgs = example["messages"]
        system_parts = [m["content"] for m in msgs if m.get("role") == "system" and m.get("content")]
        user_parts   = [m["content"] for m in msgs if m.get("role") == "user"]
        asst_parts   = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if not user_parts or not asst_parts:
            return None
        # system 内容拼到 instruction 前面
        system_text = system_parts[0].strip() if system_parts else ""
        user_text   = user_parts[0].strip()
        instruction = f"{system_text}\n{user_text}".strip() if system_text else user_text
        output      = asst_parts[0].strip()
        if instruction and output:
            return {"instruction": instruction, "input": "", "output": output}

    # ── 备用：prompt / response 字段 ──
    if "prompt" in example and "response" in example:
        instruction = (example.get("prompt") or "").strip()
        output      = (example.get("response") or "").strip()
        if instruction and output:
            return {"instruction": instruction, "input": "", "output": output}

    return None  # 无法识别，跳过


# ──────────────────────────────────────────────────────────────────────────────
# 主混合逻辑
# ──────────────────────────────────────────────────────────────────────────────
def mix(domain_file: str, general_count: int, output_file: str, seed: int):
    from datasets import load_dataset

    random.seed(seed)

    # 1. 加载领域数据（JSON 数组格式）
    print(f"[1/4] 加载领域数据: {domain_file}")
    with open(domain_file, encoding="utf-8") as f:
        domain_data = json.load(f)
    print(f"      领域数据: {len(domain_data)} 条")

    # 2. 流式采样通用数据（从多个子集各采一部分）
    print(f"[2/4] 流式采样通用数据 {general_count} 条 from {NEMOTRON_DATASET}，子集: {NEMOTRON_CONFIGS}...")
    per_config = general_count // len(NEMOTRON_CONFIGS)

    general_data = []
    skipped = 0
    for config in NEMOTRON_CONFIGS:
        print(f"      采样子集 [{config}] 目标 {per_config} 条...")
        try:
            stream = load_dataset(NEMOTRON_DATASET, config, split="train", streaming=True)
            shuffled = stream.shuffle(seed=seed, buffer_size=10000)
            count_before = len(general_data)
            for example in shuffled:
                if len(general_data) - count_before >= per_config:
                    break
                converted = convert_nemotron(example)
                if converted:
                    converted["source"] = f"nemotron_{config}"
                    general_data.append(converted)
                else:
                    skipped += 1
            print(f"        采到 {len(general_data) - count_before} 条")
        except Exception as e:
            print(f"        [跳过] {config} 加载失败: {e}")

    print(f"      通用数据合计: {len(general_data)} 条（跳过 {skipped} 条无法解析的）")

    if len(general_data) < general_count * 0.5:
        print(f"[警告] 只采样到 {len(general_data)} 条，远少于目标 {general_count} 条。")
        print("       请先运行 --inspect 查看字段名，确认 convert_nemotron() 适配正确。")

    # 3. 给领域数据加 source 标记
    for item in domain_data:
        item.setdefault("source", "sid_domain")

    # 4. 合并 & 打乱
    print(f"[3/4] 合并 & 打乱...")
    mixed = domain_data + general_data
    random.shuffle(mixed)
    print(f"      混合后总计: {len(mixed)} 条")
    print(f"      领域占比: {len(domain_data)/len(mixed):.1%}，通用占比: {len(general_data)/len(mixed):.1%}")

    # 5. 保存（JSON 数组，LlamaFactory 兼容）
    print(f"[4/4] 保存到 {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False, indent=2)
    print(f"      完成！文件: {output_file}")

    # 打印示例
    print("\n── 混合数据示例（各取1条）──")
    sid_sample = next((x for x in mixed if x.get("source") == "sid_domain"), None)
    gen_sample = next((x for x in mixed if x.get("source") == "nemotron_general"), None)
    if sid_sample:
        print(f"[领域] instruction: {sid_sample['instruction'][:100]}...")
        print(f"       output:      {sid_sample['output'][:80]}...")
    if gen_sample:
        print(f"[通用] instruction: {gen_sample['instruction'][:100]}...")
        print(f"       output:      {gen_sample['output'][:80]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="探查 Nemotron 数据集字段后退出")
    parser.add_argument("--domain_file", default="data/augment_sft.json")
    parser.add_argument("--general_count", type=int, default=9000, help="采样的通用数据条数")
    parser.add_argument("--output", default="data/mixed_sft.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.inspect:
        inspect_nemotron()
    else:
        mix(args.domain_file, args.general_count, args.output, args.seed)


if __name__ == "__main__":
    main()
