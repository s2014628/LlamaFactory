"""
SID预训练效果测试脚本
测试模型是否学会了 SID token <-> 经验轨迹 的双向映射

用法：
  python scripts/test_sid_pretrain.py \
      --model_path saves/sid_pretrain/full/Qwen-4B-20260327 \
      --data_path data/sid_augmented_pt_20260327.jsonl \
      --num_samples 20 \
      --mode both

参数：
  --mode: sid2desc (SID->描述), desc2sid (描述->SID), both (双向都测)
"""
import argparse
import json
import random
import re
import sys
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    print(f"[加载模型] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[模型加载完成] 设备: {next(model.parameters()).device}")
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy，结果可复现
            temperature=1.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成部分
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=False).strip()


def parse_sid(text: str):
    """从文本中提取 SID token 三元组，返回字符串或 None"""
    m = re.search(r"(<SID_L1_\d+>)(<SID_L2_\d+>)(<SID_L3_\d+>)", text)
    return "".join(m.groups()) if m else None


def split_sample(text: str):
    """
    根据文本格式判断是哪个方向，拆分为 (prompt前缀, 期望续写内容, 方向)
    返回: (prompt, expected, direction) 或 None
    """
    # 方向①: SID -> 描述
    # 格式: "<SID_L1_X><SID_L2_Y><SID_L3_Z> The experience trajectory for this SID is:\n..."
    m1 = re.match(
        r"(<SID_L1_\d+><SID_L2_\d+><SID_L3_\d+>)\s*(The experience trajectory for this SID is:\n)(.*)",
        text,
        re.DOTALL,
    )
    if m1:
        sid_part = m1.group(1)
        bridge = m1.group(2)
        expected = m1.group(3).strip()
        prompt = f"{sid_part} {bridge}"
        return prompt, expected, "sid2desc"

    # 方向②: 描述 -> SID
    # 格式: "The SID that may correspond to...\nCorresponding SID: <SID_...>"
    m2 = re.search(r"Corresponding SID:\s*(<SID_L1_\d+><SID_L2_\d+><SID_L3_\d+>)", text)
    if m2:
        expected_sid = m2.group(1)
        prompt = text[: m2.start()] + "Corresponding SID:"
        return prompt, expected_sid, "desc2sid"

    return None


def evaluate_sid_match(generated: str, expected_sid: str) -> bool:
    pred = parse_sid(generated)
    return pred == expected_sid


def wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(indent + line)
        else:
            for chunk in textwrap.wrap(line, width):
                wrapped.append(indent + chunk)
    return "\n".join(wrapped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saves/sid_pretrain/full/Qwen-4B-20260327")
    parser.add_argument("--data_path", default="data/sid_augmented_pt_20260327.jsonl")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--mode", choices=["sid2desc", "desc2sid", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载数据
    with open(args.data_path, encoding="utf-8") as f:
        all_samples = [json.loads(line)["text"] for line in f if line.strip()]

    print(f"[数据] 共 {len(all_samples)} 条，随机抽取 {args.num_samples} 条")

    # 按方向过滤
    parsed = []
    for text in all_samples:
        result = split_sample(text)
        if result is None:
            continue
        prompt, expected, direction = result
        if args.mode == "both" or args.mode == direction:
            parsed.append((prompt, expected, direction))

    if not parsed:
        print("[错误] 没有匹配的样本，请检查数据格式或 --mode 参数")
        sys.exit(1)

    samples = random.sample(parsed, min(args.num_samples, len(parsed)))
    print(f"[过滤后] 有效样本 {len(parsed)} 条，本次测试 {len(samples)} 条\n")

    tokenizer, model = load_model(args.model_path)

    sid2desc_correct = 0
    sid2desc_total = 0
    desc2sid_correct = 0
    desc2sid_total = 0

    sep = "=" * 100

    for idx, (prompt, expected, direction) in enumerate(samples, 1):
        print(sep)
        label = "SID → 描述" if direction == "sid2desc" else "描述 → SID"
        print(f"[样本 {idx:02d}] 方向: {label}")
        print(f"  Prompt:\n{wrap(prompt)}")

        generated = generate(tokenizer, model, prompt, args.max_new_tokens)

        if direction == "sid2desc":
            sid2desc_total += 1
            # 打印生成内容与期望内容的前200字对比
            gen_preview = generated[:300].replace("\n", " ")
            exp_preview = expected[:300].replace("\n", " ")
            print(f"  [生成]: {gen_preview}")
            print(f"  [期望]: {exp_preview}")
            # 简单评估：生成内容包含期望文本的关键片段（前50字）
            key_frag = expected[:50].strip()
            hit = key_frag.lower() in generated.lower()
            if hit:
                sid2desc_correct += 1
            print(f"  [匹配前50字]: {'✅ HIT' if hit else '❌ MISS'}")

        else:  # desc2sid
            desc2sid_total += 1
            pred_sid = parse_sid(generated)
            hit = pred_sid == expected
            if hit:
                desc2sid_correct += 1
            print(f"  [生成SID]: {pred_sid or '(未提取到SID)'}")
            print(f"  [期望SID]: {expected}")
            print(f"  [SID完全匹配]: {'✅ HIT' if hit else '❌ MISS'}")

    print(sep)
    print("\n========== 测试结果汇总 ==========")
    if sid2desc_total:
        print(f"  SID → 描述  前50字命中率: {sid2desc_correct}/{sid2desc_total} = {sid2desc_correct/sid2desc_total:.1%}")
    if desc2sid_total:
        print(f"  描述 → SID  精确匹配率:   {desc2sid_correct}/{desc2sid_total} = {desc2sid_correct/desc2sid_total:.1%}")
    print("====================================\n")

    print("提示：如果命中率很低，可能原因：")
    print("  1. 训练步数不足（当前 checkpoint 步数可在 trainer_state.json 中查看）")
    print("  2. 测试 prompt 与训练时模板格式不一致")
    print("  3. 测试数据与训练数据重叠度低（随机采样偏差）")


if __name__ == "__main__":
    main()
