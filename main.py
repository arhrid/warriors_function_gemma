
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"function_calls": [], "total_time_ms": 0}
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        return {"function_calls": [], "total_time_ms": 0}

    try:
        gemini_tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                            for k, v in t["parameters"]["properties"].items()
                        },
                        required=t["parameters"].get("required", []),
                    ),
                )
                for t in tools
            ])
        ]

        contents = [m["content"] for m in messages if m["role"] == "user"]

        start_time = time.time()

        gemini_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=gemini_tools),
        )

        total_time_ms = (time.time() - start_time) * 1000

        function_calls = []
        for candidate in gemini_response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append({
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args),
                    })

        return {
            "function_calls": function_calls,
            "total_time_ms": total_time_ms,
        }
    except Exception:
        return {"function_calls": [], "total_time_ms": 0}


def _coerce_args(arguments, tool):
    """Coerce argument values to match the tool schema types."""
    props = tool["parameters"].get("properties", {})
    coerced = {}
    for key, val in arguments.items():
        expected_type = props.get(key, {}).get("type", "string")
        if expected_type == "integer" and isinstance(val, str):
            try:
                coerced[key] = int(val)
            except ValueError:
                coerced[key] = val
        elif expected_type == "number" and isinstance(val, str):
            try:
                coerced[key] = float(val)
            except ValueError:
                coerced[key] = val
        elif expected_type == "integer" and isinstance(val, float):
            coerced[key] = int(val)
        else:
            coerced[key] = val
    return coerced


def _enhance_tools(tools):
    """Enhance tool descriptions with parameter type hints for FunctionGemma."""
    enhanced = []
    for t in tools:
        et = json.loads(json.dumps(t))  # deep copy
        props = et["parameters"].get("properties", {})
        hints = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", pname)
            if ptype == "integer":
                hints.append(f"{pname}: integer ({pdesc})")
            elif ptype == "string":
                hints.append(f"{pname}: string ({pdesc})")
            else:
                hints.append(f"{pname}: {ptype} ({pdesc})")
        if hints:
            et["description"] = et["description"] + ". Parameters: " + ", ".join(hints)
        enhanced.append(et)
    return enhanced


def _validate_calls(function_calls, tools_by_name, tool_required):
    """Filter to calls with valid tool names, required args, and coerce types."""
    valid = []
    seen_names = set()
    for fc in function_calls:
        name = fc.get("name", "")
        args = fc.get("arguments", {})
        if name in tools_by_name and name not in seen_names:
            required = tool_required.get(name, [])
            if all(r in args for r in required):
                coerced_args = _coerce_args(args, tools_by_name[name])
                valid.append({"name": name, "arguments": coerced_args})
                seen_names.add(name)
    return valid


def _run_cactus(messages, tools, system_prompt=None, tool_rag_top_k=0, enhance=True):
    """Run on-device inference with fresh model."""
    model = cactus_init(functiongemma_path)

    actual_tools = _enhance_tools(tools) if enhance else tools
    cactus_tools = [{"type": "function", "function": t} for t in actual_tools]

    if system_prompt is None:
        system_prompt = (
            "You are a function calling assistant. "
            "Call the correct function(s) with accurate arguments extracted from the user request. "
            "Use exact values from the request. "
            "If multiple actions are needed, make ALL function calls."
        )

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=tool_rag_top_k,
        confidence_threshold=0.05,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Smart hybrid routing: maximize on-device + correctness.

    Strategy: on-device with F1>=0.46 beats perfect cloud (scoring math).
    So we aggressively trust on-device when it produces structurally valid
    function calls (correct tool name + all required args + type coercion).
    Cloud only when on-device produces nothing usable.
    """
    tools_by_name = {t["name"]: t for t in tools}
    tool_required = {t["name"]: t["parameters"].get("required", []) for t in tools}

    # === Attempt 1: on-device with enhanced tool descriptions ===
    local = _run_cactus(messages, tools, tool_rag_top_k=0, enhance=True)
    valid_calls = _validate_calls(local["function_calls"], tools_by_name, tool_required)

    if valid_calls:
        return {
            "function_calls": valid_calls,
            "total_time_ms": local["total_time_ms"],
            "confidence": local["confidence"],
            "source": "on-device",
        }

    # === Attempt 2: retry without enhancement (different prompt path) ===
    local2 = _run_cactus(messages, tools, tool_rag_top_k=0, enhance=False)
    valid2 = _validate_calls(local2["function_calls"], tools_by_name, tool_required)

    if valid2:
        return {
            "function_calls": valid2,
            "total_time_ms": local["total_time_ms"] + local2["total_time_ms"],
            "confidence": local2["confidence"],
            "source": "on-device",
        }

    # === Attempt 3: cloud fallback (last resort) ===
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"] + local2["total_time_ms"]
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
