"""
test_sft_format.py

快速验证 MemRetriever SFT 模型的 ReAct 格式输出。
检查：
  1. 是否生成 <think>...</think>
  2. 是否生成 <tool_call>{"name":"memory_lookup","arguments":{"sid_list":[...]}}
  3. tool_call 里是否有合法的 SID token（<SID_L1_X><SID_L2_Y><SID_L3_Z>）
  4. 是否生成 <answer>...</answer>（在 tool_response 注入之后）

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_sft_format.py --model v2
    CUDA_VISIBLE_DEVICES=1 python test_sft_format.py --model mixed
"""

import argparse
import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    "v2":    "/data2/jxk/LlamaFactory/saves/mem_retriever_react_sft_v2/full/Qwen-8B",
    "mixed": "/data2/jxk/LlamaFactory/saves/mem_retriever_react_sft_mixed/full/Qwen-8B",
}

SYSTEM_PROMPT = (
    "You are MemRetriever, a memory retrieval agent. "
    "For each query, reason in <think>...</think>, retrieve relevant experience "
    "using memory_lookup, then summarise in <answer>...</answer>.\n\n"
    "Example:\n"
    "<think>\nThis query is about arithmetic word problems. I will try some SIDs.\n</think>\n"
    "<tool_call>\n"
    '{"name": "memory_lookup", "arguments": {"sid_list": ["<SID_L1_5><SID_L2_3><SID_L3_2>"]}}\n'
    "</tool_call>\n"
    "<tool_response>\n"
    "[SID: <SID_L1_5><SID_L2_3><SID_L3_2>] Dataset: math | "
    "Experience: To solve multi-step arithmetic: list quantities, apply operations in order.\n"
    "</tool_response>\n"
    "<think>\nFound relevant experience.\n</think>\n"
    "<answer>\nFor multi-step arithmetic: list all quantities, apply each operation in order.\n</answer>"
)

TOOL_SCHEMA = [{
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
                        "List of SID token strings. "
                        'Examples: ["<SID_L1_5><SID_L2_3><SID_L3_2>"], '
                        '["<SID_L1_12><SID_L2_0><SID_L3_7>", "<SID_L1_3><SID_L2_15><SID_L3_1>"]. '
                        "Do NOT use natural-language descriptions as SID values."
                    ),
                }
            },
            "required": ["sid_list"],
        },
    },
}]

TEST_QUERIES = [
    ("math",      "What is the best strategy to solve a multi-step arithmetic word problem?"),
    ("hotpotqa",  "Who was the director of the movie that won the Academy Award for Best Picture in 2020?"),
    ("nq",        "What is the capital city of Australia?"),
    ("triviaqa",  "Which element has the atomic number 79?"),
    ("musique",   "Who is the spouse of the person who founded the company that makes the iPhone?"),
]

SID_RE = re.compile(r"<SID_L1_\d+><SID_L2_\d+><SID_L3_\d+>")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# ── fake tool_response injected after model stops at </tool_call> ─────────────
FAKE_TOOL_RESPONSE = (
    "<tool_response>\n"
    "[SID: <SID_L1_5><SID_L2_3><SID_L3_2>] Dataset: math | "
    "Experience: Break the problem into steps; identify what each quantity represents.\n"
    "</tool_response>\n"
)


def check_turn1(text: str) -> dict:
    has_think  = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
    tc_match   = TOOL_CALL_RE.search(text)
    has_tc     = tc_match is not None
    sid_in_tc  = False
    sid_tokens = []
    if tc_match:
        try:
            obj = json.loads(tc_match.group(1))
            sids = obj.get("arguments", {}).get("sid_list", [])
            for s in sids:
                if SID_RE.match(s):
                    sid_in_tc = True
                    sid_tokens.append(s)
        except Exception:
            pass
    return {"has_think": has_think, "has_tool_call": has_tc,
            "sid_in_tc": sid_in_tc, "sid_tokens": sid_tokens, "raw": text}


def check_turn2(text: str) -> dict:
    has_think  = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
    return {"has_think": has_think, "has_answer": has_answer, "raw": text}


def run_turn(model, tokenizer, messages, tools, max_new=256, stop_str="</tool_call>"):
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    # truncate at stop string if present
    if stop_str in generated:
        generated = generated[:generated.index(stop_str) + len(stop_str)]
    return generated.strip()


def test_model(model_name: str):
    path = MODEL_PATHS[model_name]
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({path})")
    print(f"{'='*60}")

    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    results = []
    for domain, query in TEST_QUERIES:
        print(f"\n[{domain}] {query[:60]}...")

        # ── Turn 1: model generates think + tool_call ──────────────────
        msgs = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": f"<query> {query} </query>"},
        ]
        gen1 = run_turn(model, tokenizer, msgs, TOOL_SCHEMA, max_new=256)
        r1   = check_turn1(gen1)
        print(f"  turn1 → think:{r1['has_think']} | tool_call:{r1['has_tool_call']} "
              f"| SID_in_tc:{r1['sid_in_tc']} | SIDs:{r1['sid_tokens']}")
        print(f"  raw1 : {gen1[:200].replace(chr(10),' ')}")

        # ── Turn 2: inject tool_response, model generates think + answer ─
        msgs2 = msgs + [
            {"role": "assistant", "content": gen1},
            {"role": "tool",      "content": FAKE_TOOL_RESPONSE},
        ]
        gen2 = run_turn(model, tokenizer, msgs2, TOOL_SCHEMA, max_new=256, stop_str="</answer>")
        r2   = check_turn2(gen2)
        print(f"  turn2 → think:{r2['has_think']} | answer:{r2['has_answer']}")
        print(f"  raw2 : {gen2[:200].replace(chr(10),' ')}")

        results.append({
            "domain": domain,
            "turn1": r1,
            "turn2": r2,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SUMMARY [{model_name}]")
    t1_ok = sum(1 for r in results if r["turn1"]["sid_in_tc"])
    t2_ok = sum(1 for r in results if r["turn2"]["has_answer"])
    print(f"  Turn1 SID-in-tool_call : {t1_ok}/{len(results)}")
    print(f"  Turn2 <answer> present : {t2_ok}/{len(results)}")
    print(f"{'='*60}\n")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["v2", "mixed", "both"], default="both")
    args = parser.parse_args()

    targets = ["v2", "mixed"] if args.model == "both" else [args.model]
    for m in targets:
        test_model(m)
