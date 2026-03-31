"""
将 memory_bank.json 转换为双向 SFT 格式：
  - 正向：experience_summary/text → SID token
  - 反向：SID token → experience_summary/text
"""

import json
import random
from pathlib import Path

INPUT_PATH = "/data2/jxk/RQ-KMeans/rqkmeans_8k_512_256_128/memory_bank.json"
OUTPUT_PATH = "/data2/jxk/LlamaFactory/data/memory_bank_sft.json"

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
    """优先使用 experience_summary，否则退回 experience_text。"""
    summary = str(sample.get("experience_summary", "")).strip()
    if summary:
        return summary
    return str(sample.get("experience_text", "")).strip()


def convert(data: list[dict]) -> list[dict]:
    results = []
    for sample in data:
        sid_token = sample.get("sid", {}).get("tokens", "")
        if not sid_token:
            continue

        desc = get_desc(sample)
        if not desc:
            continue

        # 正向：desc → SID
        fwd_instr = random.choice(FORWARD_INSTRUCTIONS).format(desc=desc)
        results.append({"instruction": fwd_instr, "input": "", "output": sid_token})

        # 反向：SID → desc
        bwd_instr = random.choice(BACKWARD_INSTRUCTIONS).format(sid=sid_token)
        results.append({"instruction": bwd_instr, "input": "", "output": desc})

    return results


def main():
    print(f"读取 {INPUT_PATH} ...")
    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"原始样本数: {len(data)}")

    random.seed(42)
    sft_data = convert(data)
    print(f"转换后 SFT 条数: {len(sft_data)}（正反向各 {len(sft_data)//2} 条）")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {OUTPUT_PATH}")

    # 打印示例
    print("\n=== 正向示例 ===")
    print("instruction:", sft_data[0]["instruction"][:200])
    print("output:", sft_data[0]["output"])
    print("\n=== 反向示例 ===")
    print("instruction:", sft_data[1]["instruction"])
    print("output:", sft_data[1]["output"][:200])


if __name__ == "__main__":
    main()
