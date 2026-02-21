
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
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


def _extract_numbers(text):
    """Extract all numbers from user text for integer arg correction."""
    return [int(x) for x in re.findall(r'\b(\d+)\b', text)]


def _extract_time(text):
    """Extract hour/minute from time patterns like '10 AM', '3:00 PM', '8:15 AM'."""
    match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3).upper().replace('.', '')
        if period == "PM" and hour < 12:
            hour += 12
        elif period == "AM" and hour == 12:
            hour = 0
        return hour, minute
    return None, None


def _fix_integer_args(call, tool, user_text):
    """Post-process integer arguments using values extracted from user text."""
    props = tool["parameters"].get("properties", {})
    args = dict(call.get("arguments", {}))

    # Check if this tool has hour/minute params (time-related)
    param_names = set(props.keys())
    if "hour" in param_names and "minute" in param_names:
        hour, minute = _extract_time(user_text)
        if hour is not None:
            args["hour"] = hour
            args["minute"] = minute
            return args

    # For single-integer params, extract from user text
    int_params = [k for k, v in props.items() if v.get("type") == "integer"]
    if len(int_params) == 1:
        numbers = _extract_numbers(user_text)
        if len(numbers) == 1:
            args[int_params[0]] = numbers[0]

    return args


def _is_multi_request(text):
    """Detect if user message likely contains multiple requests."""
    lower = text.lower()
    if " and " not in lower:
        return False
    parts = re.split(r'\s+and\s+', lower)
    if len(parts) < 2:
        return False
    action_words = {"set", "get", "send", "check", "find", "look", "play",
                    "remind", "text", "search", "create", "what", "how"}
    action_parts = sum(1 for p in parts
                       if any(p.strip().startswith(w) for w in action_words))
    return action_parts >= 2


def _is_tool_relevant(tool, user_text):
    """Check if a tool is semantically relevant to the user text."""
    desc_words = set(re.findall(r'\w+', tool["description"].lower()))
    user_words = set(re.findall(r'\w+', user_text.lower()))
    # Also include tool name words
    name_words = set(tool["name"].lower().replace("_", " ").split())
    tool_words = desc_words | name_words
    # Check param description words too
    for p in tool["parameters"].get("properties", {}).values():
        pdesc = p.get("description", "").lower()
        tool_words.update(re.findall(r'\w+', pdesc))
    overlap = tool_words & user_words
    # Filter common stopwords from overlap
    stopwords = {"a", "an", "the", "for", "to", "of", "in", "is", "and", "or", "it"}
    meaningful_overlap = overlap - stopwords
    return len(meaningful_overlap) >= 1


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Smart hybrid routing with per-tool completion and arg correction.

    Strategies:
    1. Enhanced tool descriptions for better FunctionGemma accuracy.
    2. Integer arg post-processing from user text (fixes FunctionGemma's
       common mistake of hallucinating integer values).
    3. Per-tool single-call completion for multi-request messages:
       after initial call, run each remaining relevant tool individually
       (1 tool + force_tools = can't pick wrong tool).
    4. Cloud fallback only when on-device produces nothing valid.
    """
    tools_by_name = {t["name"]: t for t in tools}
    tool_required = {t["name"]: t["parameters"].get("required", []) for t in tools}
    user_text = messages[-1]["content"] if messages else ""
    total_time = 0

    # === Phase 1: on-device with all tools ===
    local = _run_cactus(messages, tools, tool_rag_top_k=0, enhance=True)
    total_time += local["total_time_ms"]
    valid_calls = _validate_calls(local["function_calls"], tools_by_name, tool_required)

    # Fix integer arguments
    for i, call in enumerate(valid_calls):
        tool = tools_by_name.get(call["name"])
        if tool:
            valid_calls[i]["arguments"] = _fix_integer_args(call, tool, user_text)

    # === Phase 1b: retry without enhancement if nothing valid ===
    if not valid_calls:
        local2 = _run_cactus(messages, tools, tool_rag_top_k=0, enhance=False)
        total_time += local2["total_time_ms"]
        valid_calls = _validate_calls(local2["function_calls"], tools_by_name, tool_required)
        for i, call in enumerate(valid_calls):
            tool = tools_by_name.get(call["name"])
            if tool:
                valid_calls[i]["arguments"] = _fix_integer_args(call, tool, user_text)

    # === Phase 2: per-tool completion for multi-request messages ===
    # When user asks for multiple things, FunctionGemma often returns only 1 call.
    # For each remaining RELEVANT tool, run a single-tool cactus call.
    # With 1 tool + force_tools, the model can't pick the wrong tool.
    if valid_calls and _is_multi_request(user_text):
        called_names = {c["name"] for c in valid_calls}
        for t in tools:
            if t["name"] in called_names:
                continue
            if not _is_tool_relevant(t, user_text):
                continue
            single = _run_cactus(messages, [t], tool_rag_top_k=0, enhance=True)
            total_time += single["total_time_ms"]
            extra = _validate_calls(single["function_calls"], tools_by_name, tool_required)
            for call in extra:
                if call["name"] not in called_names:
                    tool_def = tools_by_name.get(call["name"])
                    if tool_def:
                        call["arguments"] = _fix_integer_args(call, tool_def, user_text)
                    valid_calls.append(call)
                    called_names.add(call["name"])

    # === Return on-device if valid ===
    if valid_calls:
        return {
            "function_calls": valid_calls,
            "total_time_ms": total_time,
            "confidence": local["confidence"],
            "source": "on-device",
        }

    # === Phase 3: cloud fallback ===
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += total_time
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
