"""
build_mem_retriever_sft.py

从 memory bank 构造 MemRetriever 多轮 SFT 数据。
格式：query → think → tool_call(正确SID) → tool_response(经验) → think → answer

Usage:
    python build_mem_retriever_sft.py
"""

import json
import random
import re

MEMORY_BANK_PATH = "/data2/jxk/RQ-KMeans/rqkmeans_8k_64_32_16/memory_bank.json"
OUTPUT_PATH      = "/data2/jxk/LlamaFactory/data/mem_retriever_react_sft_v2.json"
TRAIN_RATIO      = 0.95
SEED             = 42
MAX_SAMPLES      = 20000   # 上限，够了就截断

TOOL_SCHEMA_STR = json.dumps([{
    "type": "function",
    "function": {
        "name": "memory_lookup",
        "description": (
            "Look up experience memories from the memory bank using Semantic IDs (SIDs). "
            "Returns the top-k most relevant experience summaries for each SID provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sid_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        'List of SID token strings. '
                        'Examples: ["<SID_L1_5><SID_L2_3><SID_L3_2>"], '
                        '["<SID_L1_12><SID_L2_0><SID_L3_7>", "<SID_L1_3><SID_L2_15><SID_L3_1>"]. '
                        'Do NOT use natural-language descriptions as SID values.'
                    ),
                }
            },
            "required": ["sid_list"],
        },
    },
}], ensure_ascii=False)

SYSTEM_PROMPT = (
    "You are MemRetriever, a memory retrieval agent. "
    "For each query, reason in <think>...</think>, retrieve relevant experience "
    "using memory_lookup, then summarise in <answer>...</answer>."
)

# 简单的 think 模板，目的是教格式而非推理质量
THINK1_TEMPLATES = [
    "This question involves {domain}. I will look up relevant memories using SID tokens.",
    "The query is about {domain}. Let me retrieve related experience from the memory bank.",
    "I need to find memories related to {domain}. I'll call memory_lookup with the appropriate SID.",
    "This is a {domain} question. I'll search the memory bank for relevant experience.",
]

THINK2_TEMPLATES = [
    "I found relevant experience. Let me organise and summarise it.",
    "The retrieved experience is relevant to the query. I'll summarise the key points.",
    "Good, I have the relevant experience. Let me provide a clean summary.",
    "The memory lookup returned useful information. I'll synthesise it into a clear answer.",
]

DATASET_DOMAIN = {
    "math":      "mathematics and arithmetic",
    "popqa":     "factual knowledge lookup",
    "triviaqa":  "trivia and general knowledge",
    "hotpotqa":  "multi-hop reasoning",
    "nq":        "general knowledge",
    "musique":   "multi-step reasoning",
    "2wikimqa":  "multi-hop knowledge",
    "default":   "general problem solving",
}


def get_domain(dataset: str) -> str:
    for key, val in DATASET_DOMAIN.items():
        if key in dataset.lower():
            return val
    return DATASET_DOMAIN["default"]


def build_sample(entry: dict, idx: int) -> dict | None:
    question = entry.get("question", "").strip()
    sid_info  = entry.get("sid", {})
    sid_tokens = sid_info.get("tokens", "")
    exp_text   = (entry.get("experience_summary") or entry.get("experience_text", "")).strip()
    dataset    = entry.get("dataset", "default")

    if not question or not sid_tokens or not exp_text:
        return None
    # SID格式校验
    if not re.match(r"<SID_L1_\d+><SID_L2_\d+><SID_L3_\d+>", sid_tokens):
        return None

    domain = get_domain(dataset)
    rng    = random.Random(idx)

    think1 = rng.choice(THINK1_TEMPLATES).format(domain=domain)
    think2 = rng.choice(THINK2_TEMPLATES)

    tool_call_json = json.dumps(
        {"name": "memory_lookup", "arguments": {"sid_list": [sid_tokens]}},
        ensure_ascii=False,
    )

    tool_response = (
        f"[SID: {sid_tokens}] Dataset: {dataset} | Experience: {exp_text}"
    )

    # 多轮对话：human → gpt(think+tool_call) → observation(tool_response) → gpt(think+answer)
    # tool_call_json 也作为 function_call turn，让LlamaFactory用原生function call格式
    conversations = [
        {
            "from": "system",
            "value": SYSTEM_PROMPT,
        },
        {
            "from": "human",
            "value": f"<query> {question} </query>",
        },
        {
            "from": "gpt",
            "value": (
                f"<think>\n{think1}\n</think>\n"
                f"<tool_call>\n{tool_call_json}\n</tool_call>"
            ),
        },
        {
            "from": "observation",
            "value": tool_response,
        },
        {
            "from": "gpt",
            "value": (
                f"<think>\n{think2}\n</think>\n"
                f"<answer>\n{exp_text}\n</answer>"
            ),
        },
    ]

    return {"conversations": conversations, "tools": TOOL_SCHEMA_STR}


def main():
    print(f"Loading memory bank from {MEMORY_BANK_PATH} ...")
    with open(MEMORY_BANK_PATH, encoding="utf-8") as f:
        bank = json.load(f)

    random.seed(SEED)
    random.shuffle(bank)

    samples = []
    for i, entry in enumerate(bank):
        if len(samples) >= MAX_SAMPLES:
            break
        s = build_sample(entry, i)
        if s:
            samples.append(s)

    print(f"Built {len(samples)} samples from {len(bank)} entries")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
