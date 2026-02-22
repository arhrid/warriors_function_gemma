import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json, re
from datasets import load_dataset
from main import generate_hybrid

# ── Load Glaive ───────────────────────────────────────────────
print("Loading Glaive dataset...")
ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

RELEVANT_KEYWORDS = [
    "weather", "alarm", "message", "reminder",
    "contacts", "music", "timer", "send", "play"
]

def extract_sample(sample):
    try:
        system = sample.get("system", "")
        chat   = sample.get("chat", "")

        # Tools are individual {…} objects with nested braces — use JSON decoder
        tools = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(system):
            if system[idx] == '{':
                try:
                    obj, end = decoder.raw_decode(system, idx)
                    if isinstance(obj, dict) and "name" in obj:
                        tools.append(obj)
                    idx = end
                except json.JSONDecodeError:
                    idx += 1
            else:
                idx += 1
        if not tools:
            return None

        tool_desc = " ".join(
            t.get("description", "") for t in tools
        ).lower()
        if not any(kw in tool_desc for kw in RELEVANT_KEYWORDS):
            return None

        user_match = re.search(
            r'USER:\s*(.+?)(?:ASSISTANT:|$)', chat, re.DOTALL)
        if not user_match:
            return None
        user_msg = user_match.group(1).strip()

        # Extract function call — Glaive uses single-quoted arguments strings
        fc_pos = chat.find("<functioncall>")
        if fc_pos < 0:
            return None
        fc_text = chat[fc_pos + len("<functioncall>"):]
        # Replace single-quoted arguments with inline JSON
        fc_text = re.sub(r'"arguments"\s*:\s*\'(.+?)\'', r'"arguments": \1', fc_text)
        fc_text = re.sub(r"'arguments'\s*:\s*'(.+?)'", r'"arguments": \1', fc_text)
        brace_pos = fc_text.find("{")
        if brace_pos < 0:
            return None
        try:
            call, _ = json.JSONDecoder().raw_decode(fc_text, brace_pos)
        except json.JSONDecodeError:
            return None

        # Convert tools to your format
        converted_tools = []
        for t in tools:
            params = t.get("parameters", {})
            props  = params.get("properties", {})
            converted_tools.append({
                "name":        t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {
                            "type":        v.get("type", "string"),
                            "description": v.get("description", k)
                        }
                        for k, v in props.items()
                    },
                    "required": params.get("required", [])
                }
            })

        return {
            "user_msg": user_msg,
            "tools":    converted_tools,
            "expected_name": call["name"],
            "expected_args": call.get("arguments", {})
        }
    except:
        return None


# ── Run generate_hybrid ───────────────────────────────────────
def run_on_glaive(max_cases=20):
    cases = []
    print("Filtering relevant cases...")
    for sample in ds.select(range(5000)):
        extracted = extract_sample(sample)
        if extracted:
            cases.append(extracted)
        if len(cases) >= max_cases:
            break

    print(f"Found {len(cases)} relevant cases\n")
    print(f"{'#':>3} | {'Time':>8} | {'Source':<16} | {'Name':>6} | Query")
    print("-" * 85)

    name_correct  = 0
    total_time_ms = 0

    for i, case in enumerate(cases, 1):
        messages = [{"role": "user", "content": case["user_msg"]}]
        result   = generate_hybrid(messages, case["tools"])

        predicted_names = [c["name"] for c in result["function_calls"]]
        name_matched    = case["expected_name"] in predicted_names

        if name_matched:
            name_correct += 1

        source     = result.get("source", "unknown")
        time_ms    = result.get("total_time_ms", 0)
        total_time_ms += time_ms
        query      = case["user_msg"][:45].replace("\n", " ")

        print(f"{i:>3} | {time_ms:>7.0f}ms | {source:<16} | "
              f"{'✅' if name_matched else '❌':<6} | {query}")

    n = len(cases)
    on_device_count = sum(
        1 for c in cases
        if generate_hybrid(
            [{"role": "user", "content": c["user_msg"]}],
            c["tools"]
        ).get("source") == "on-device"
    )

    print("-" * 85)
    print(f"\n📊 Results on {n} Glaive cases:")
    if n > 0:
        print(f"   Tool name accuracy : {name_correct}/{n} ({100*name_correct/n:.0f}%)")
        print(f"   Avg time           : {total_time_ms/n:.0f}ms")
    else:
        print("   No relevant cases found.")


if __name__ == "__main__":
    run_on_glaive(max_cases=20)