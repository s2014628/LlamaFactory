"""
SID SFT 效果评测脚本
测试模型双向映射能力：
  - 正向：experience_summary → SID token（精确匹配）
  - 反向：SID token → experience_summary（关键词命中率）

数据源：原始 memory_bank.json（而非训练用的 SFT 格式文件），
每条样本用固定 instruction 模板构造 prompt，相当于轻微 distribution shift 测试。

用法：
  CUDA_VISIBLE_DEVICES=0 python scripts/test_sid_sft.py \
      --model_path saves/sid_sft/full/Qwen-8B-20260331 \
      --data_path /data2/jxk/RQ-KMeans/rqkmeans_8k_64_32_16/memory_bank.json \
      --num_samples 200 \
      --mode both
"""
import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 固定使用单一 instruction 模板（不在训练模板中，轻微 distribution shift）
FORWARD_INSTRUCTION = (
    "Given the following experience summary, identify the corresponding SID token. "
    "Output only the SID token in the format <SID_L1_X><SID_L2_Y><SID_L3_Z>, nothing else.\n\n"
    "{desc}"
)
BACKWARD_INSTRUCTION = (
    "The SID token is {sid}. Describe the problem-solving experience it represents."
)


def get_desc(sample: dict) -> str:
    summary = str(sample.get("experience_summary", "")).strip()
    return summary if summary else str(sample.get("experience_text", "")).strip()


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
    print(f"[模型加载完成]")
    return tokenizer, model


def generate(tokenizer, model, instruction: str, max_new_tokens: int = 64) -> str:
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=False).strip()


def parse_sid(text: str):
    """从生成文本中提取 SID 三元组，返回完整 token 字符串或 None。"""
    m = re.search(r"(<SID_L1_\d+>)(<SID_L2_\d+>)(<SID_L3_\d+>)", text)
    return "".join(m.groups()) if m else None


def strip_think(text: str) -> str:
    """去除 Qwen3 thinking 模式产生的 <think>...</think> 前缀。"""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def keyword_overlap(generated: str, expected: str, top_k: int = 10) -> float:
    """
    关键词重叠率：
    取 expected 中最长的 top_k 个词，计算在 generated 中出现的比例。
    过滤掉常见停用词和 markdown 符号。
    """
    STOPWORDS = {
        "the", "a", "an", "of", "to", "in", "is", "are", "and", "or",
        "for", "by", "with", "that", "this", "it", "its", "be", "as",
        "on", "at", "from", "into", "when", "where", "which", "how",
        "problem", "addressed", "practice", "step", "steps",
    }
    tokens = re.findall(r"[a-zA-Z]{3,}", expected.lower())
    keywords = [t for t in tokens if t not in STOPWORDS]
    # 取出现频率最高（先去重保留顺序）的 top_k 个
    seen, unique = set(), []
    for t in keywords:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    keywords = unique[:top_k]
    if not keywords:
        return 0.0
    gen_lower = generated.lower()
    hits = sum(1 for kw in keywords if kw in gen_lower)
    return hits / len(keywords)


SID_PATTERN = re.compile(r"^<SID_L1_\d+><SID_L2_\d+><SID_L3_\d+>$")


def is_forward(sample: dict) -> bool:
    """output 是纯 SID token 的为正向（desc→SID）。"""
    return bool(SID_PATTERN.match(sample["output"].strip()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saves/sid_sft/full/Qwen-8B-20260331")
    parser.add_argument(
        "--data_path",
        default="/data2/jxk/RQ-KMeans/rqkmeans_8k_64_32_16/memory_bank.json",
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument(
        "--mode", choices=["forward", "backward", "both"], default="both"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="eval_results",
        help="保存评测结果 JSON 的目录",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"[读取数据] {args.data_path}")
    with open(args.data_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # 过滤掉没有 SID 或描述的样本
    valid = [d for d in raw_data if d.get("sid", {}).get("tokens") and get_desc(d)]
    print(f"  有效样本: {len(valid)} 条")

    n_fwd = args.num_samples // 2 if args.mode == "both" else args.num_samples
    n_bwd = args.num_samples // 2 if args.mode == "both" else args.num_samples

    samples: list[tuple[str, dict]] = []
    if args.mode in ("forward", "both"):
        picked = random.sample(valid, min(n_fwd, len(valid)))
        samples += [("forward", d) for d in picked]
    if args.mode in ("backward", "both"):
        picked = random.sample(valid, min(n_bwd, len(valid)))
        samples += [("backward", d) for d in picked]

    random.shuffle(samples)
    print(f"  本次测试: {len(samples)} 条\n")

    tokenizer, model = load_model(args.model_path)

    fwd_correct, fwd_total = 0, 0
    fwd_l1_correct, fwd_l1l2_correct = 0, 0
    bwd_hit, bwd_total, bwd_score_sum = 0, 0, 0.0
    sep = "=" * 100
    records: list[dict] = []  # 每条样本的详细结果

    for idx, (direction, d) in enumerate(samples, 1):
        sid_token = d["sid"]["tokens"]
        desc = get_desc(d)
        if direction == "forward":
            instruction = FORWARD_INSTRUCTION.format(desc=desc)
            expected = sid_token
        else:
            instruction = BACKWARD_INSTRUCTION.format(sid=sid_token)
            expected = desc

        if direction == "forward":
            generated = generate(tokenizer, model, instruction, max_new_tokens=16)
            fwd_total += 1
            pred_sid = parse_sid(generated)
            hit = pred_sid == expected
            fwd_correct += hit

            # L1 / L1+L2 部分匹配
            exp_parts = re.findall(r"<SID_L\d+_\d+>", expected)
            pred_parts = re.findall(r"<SID_L\d+_\d+>", pred_sid) if pred_sid else []
            l1_ok = len(pred_parts) >= 1 and pred_parts[0] == exp_parts[0]
            l1l2_ok = len(pred_parts) >= 2 and pred_parts[:2] == exp_parts[:2]
            fwd_l1_correct += l1_ok
            fwd_l1l2_correct += l1l2_ok

            records.append({
                "idx": idx,
                "direction": "forward",
                "sample_id": d.get("sample_id", ""),
                "dataset": d.get("dataset", ""),
                "instruction": instruction,
                "expected": expected,
                "generated_raw": generated,
                "pred_sid": pred_sid,
                "hit_exact": hit,
                "hit_l1": l1_ok,
                "hit_l1l2": l1l2_ok,
            })

            print(sep)
            print(f"[{idx:03d}] 正向（desc→SID）  {'✅全对' if hit else ('⚡L1+L2对' if l1l2_ok else ('⚡L1对' if l1_ok else '❌'))}")
            print(f"  Instruction: {instruction[:120].replace(chr(10), ' ')}")
            print(f"  期望 SID: {expected}")
            print(f"  生成结果: {pred_sid or repr(generated[:80])}")

        else:
            generated = generate(tokenizer, model, instruction, max_new_tokens=300)
            bwd_total += 1

            # 去除 <think>...</think> 后再评估
            gen_clean = strip_think(generated)
            score = keyword_overlap(gen_clean, expected, top_k=10)
            # 关键词重叠率 >= 0.5 视为命中
            hit = score >= 0.5
            bwd_hit += hit
            bwd_score_sum += score

            records.append({
                "idx": idx,
                "direction": "backward",
                "sample_id": d.get("sample_id", ""),
                "dataset": d.get("dataset", ""),
                "instruction": instruction,
                "expected": expected,
                "generated_raw": generated,
                "generated_clean": gen_clean,
                "keyword_overlap": round(score, 4),
                "hit": hit,
            })

            print(sep)
            print(f"[{idx:03d}] 反向（SID→desc）  {'✅' if hit else '❌'} (关键词重叠={score:.0%})")
            print(f"  Instruction: {instruction}")
            print(f"  期望(前80字): {expected[:80]}")
            print(f"  生成(去think): {gen_clean[:80]}")

    print(sep)
    print("\n========== 测试结果汇总 ==========")
    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "num_samples": len(samples),
        "mode": args.mode,
        "seed": args.seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if fwd_total:
        summary["forward"] = {
            "total": fwd_total,
            "exact_match": fwd_correct,
            "exact_match_rate": round(fwd_correct / fwd_total, 4),
            "l1l2_match": fwd_l1l2_correct,
            "l1l2_match_rate": round(fwd_l1l2_correct / fwd_total, 4),
            "l1_match": fwd_l1_correct,
            "l1_match_rate": round(fwd_l1_correct / fwd_total, 4),
        }
        print(f"  正向 desc→SID  精确匹配(L1+L2+L3): {fwd_correct}/{fwd_total} = {fwd_correct/fwd_total:.1%}")
        print(f"  正向 desc→SID  L1+L2 匹配:         {fwd_l1l2_correct}/{fwd_total} = {fwd_l1l2_correct/fwd_total:.1%}")
        print(f"  正向 desc→SID  L1 匹配:             {fwd_l1_correct}/{fwd_total} = {fwd_l1_correct/fwd_total:.1%}")
    if bwd_total:
        summary["backward"] = {
            "total": bwd_total,
            "keyword_hit": bwd_hit,
            "keyword_hit_rate": round(bwd_hit / bwd_total, 4),
            "avg_overlap": round(bwd_score_sum / bwd_total, 4),
        }
        print(
            f"  反向 SID→desc  关键词重叠≥50%: {bwd_hit}/{bwd_total}"
            f" = {bwd_hit/bwd_total:.1%}"
            f"  (平均重叠率={bwd_score_sum/bwd_total:.1%})"
        )
    print("====================================\n")

    # 保存结果 JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = Path(args.model_path).name
    out_path = out_dir / f"eval_{model_tag}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"[结果已保存] {out_path}")


if __name__ == "__main__":
    main()
