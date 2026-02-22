import sys, os

# Resolve paths relative to this file so they work from any CWD
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, os.path.join(_parent_dir, "cactus", "python", "src"))
functiongemma_path = os.path.join(_parent_dir, "cactus", "weights", "functiongemma-270m-it")

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy

# Lazy-load google.genai to avoid subprocess import at module load time
genai = None
types = None


def _ensure_genai():
    global genai, types
    if genai is None:
        from google import genai as _genai
        from google.genai import types as _types
        genai = _genai
        types = _types


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
    _ensure_genai()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Fast-path + self-consistency + recursive decomposition hybrid inference.

    Strategy:
    1. Inject schema-derived type hints into every on-device call (no
       hardcoded domain rules — generated entirely from the tools schema).
    2. Fast path: single on-device run. If valid AND high-confidence ->
       return immediately (avoids extra samples for clear, easy cases).
    3. Self-consistency: run N-1 more times, reusing the first run.
       Argument-level majority voting: vote per argument independently
       after reaching tool-name consensus (more signal than exact match).
    4. Recursive decomposition of compound requests.
    5. Cloud fallback.
    """
    import re
    from collections import Counter

    N_SAMPLES = 3

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _make_schema_hints(tool_list):
        """
        Build a concise type-constraint reminder derived purely from the
        JSON schema of the provided tools. No hardcoded domain knowledge.
        """
        hints = []
        for t in tool_list:
            props = t.get("parameters", {}).get("properties", {})
            for field, spec in props.items():
                ftype = spec.get("type", "")
                if ftype in ("integer", "number", "boolean", "array"):
                    hints.append(f"'{t['name']}.{field}' must be {ftype}")
        if not hints:
            return None
        return (
            "Strict schema types — " + "; ".join(hints) +
            ". Never use a string where a number or boolean is required."
        )

    def _augment_messages(msgs, tool_list):
        """Prepend a schema-derived type hint as an extra system message."""
        hint = _make_schema_hints(tool_list)
        if hint:
            return [{"role": "system", "content": hint}] + msgs
        return msgs

    def _coerce(value, expected_type):
        if expected_type == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            try:
                return int(str(value).strip())
            except (ValueError, TypeError):
                return value
        if expected_type == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            try:
                return float(str(value).strip())
            except (ValueError, TypeError):
                return value
        if expected_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1")
            return bool(value)
        if expected_type == "string":
            return value if isinstance(value, str) else str(value)
        if expected_type == "array":
            if isinstance(value, list):
                return value
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [value]
            except Exception:
                return [value]
        return value

    def _validate(calls, tool_list):
        """
        Validate and type-coerce calls against tool schemas.
        Rejects calls where coercion fails (wrong type that can't be fixed).
        No arbitrary numeric range limits — self-consistency handles outliers.
        Returns (corrected, is_valid).
        """
        if not calls:
            return calls, False
        tool_map = {t["name"]: t for t in tool_list}
        corrected = []
        for call in calls:
            name = call.get("name")
            if name not in tool_map:
                return calls, False
            schema = tool_map[name].get("parameters", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            args = dict(call.get("arguments", {}))
            for field in required:
                if field not in args:
                    return calls, False
            corrected_args = {}
            for k, v in args.items():
                if k in props:
                    expected = props[k]["type"]
                    coerced = _coerce(v, expected)
                    # Reject only genuine coercion failures (type still wrong after attempt)
                    if expected == "integer" and not (isinstance(coerced, int) and not isinstance(coerced, bool)):
                        return calls, False
                    if expected == "number" and not isinstance(coerced, (int, float)):
                        return calls, False
                    corrected_args[k] = coerced
                else:
                    corrected_args[k] = v
            corrected.append({"name": name, "arguments": corrected_args})
        return corrected, True

    def _vote_calls(valid_runs, n):
        """
        Argument-level majority voting across valid runs.
        1. Find the majority tool-name sequence.
        2. Among agreeing runs, vote per argument independently.
        This is strictly more informative than requiring exact full-call match.
        Returns (voted_calls, best_confidence) or (None, 0).
        """
        if not valid_runs:
            return None, 0

        # Step 1: majority on ordered tool-name sequence
        name_seqs = [tuple(c["name"] for c in r[0]) for r in valid_runs]
        counts = Counter(name_seqs)
        best_seq, count = counts.most_common(1)[0]

        if count < max(2, n // 2 + 1):
            return None, 0

        # Step 2: runs that agree on the tool sequence
        agreeing = [r for r in valid_runs if tuple(c["name"] for c in r[0]) == best_seq]

        # Step 3: per-argument majority vote at each call position
        result_calls = []
        for i, name in enumerate(best_seq):
            args_list = [r[0][i]["arguments"] for r in agreeing]
            all_keys = set(k for args in args_list for k in args)
            voted_args = {}
            for key in all_keys:
                values = [args[key] for args in args_list if key in args]
                if not values:
                    continue
                # Vote by string representation; recover the original typed value
                val_counts = Counter(str(v) for v in values)
                winning_str = val_counts.most_common(1)[0][0]
                voted_args[key] = next(v for v in values if str(v) == winning_str)
            result_calls.append({"name": name, "arguments": voted_args})

        best_conf = max(r[1] for r in agreeing)
        return result_calls, best_conf

    def _self_consistent_with_first(first_run, msgs, tool_list, n=N_SAMPLES):
        """
        Reuse first_run, run n-1 more times with schema hints, apply
        argument-level voting. Returns (result_or_None, total_time_ms_spent).
        """
        corrected0, is_valid0 = _validate(first_run["function_calls"], tool_list)
        valid_runs = [(corrected0, first_run["confidence"])] if is_valid0 else []
        spent_time = first_run["total_time_ms"]

        aug_msgs = _augment_messages(msgs, tool_list)
        for _ in range(n - 1):
            local = generate_cactus(aug_msgs, tool_list)
            spent_time += local["total_time_ms"]
            corrected, is_valid = _validate(local["function_calls"], tool_list)
            if is_valid:
                valid_runs.append((corrected, local["confidence"]))

        voted_calls, best_conf = _vote_calls(valid_runs, n)
        if voted_calls is not None:
            return {"function_calls": voted_calls, "confidence": best_conf,
                    "total_time_ms": spent_time}, spent_time
        return None, spent_time

    def _on_device(msgs, tool_list):
        """
        Fast-path + self-consistency for a single request.
        Returns (result_or_None, total_time_ms_spent).
        """
        aug_msgs = _augment_messages(msgs, tool_list)
        first = generate_cactus(aug_msgs, tool_list)
        corrected, is_valid = _validate(first["function_calls"], tool_list)

        # Fast path: valid + high confidence -> return after 1 sample.
        if is_valid and first["confidence"] >= confidence_threshold:
            return {"function_calls": corrected, "confidence": first["confidence"],
                    "total_time_ms": first["total_time_ms"]}, first["total_time_ms"]

        # Self-consistency: run N-1 more, reuse first.
        return _self_consistent_with_first(first, msgs, tool_list)

    _VERBS = {
        "set", "send", "play", "get", "find", "search", "look", "check",
        "create", "make", "add", "call", "text", "remind", "show", "turn",
        "open", "close", "start", "stop", "enable", "disable", "run",
        "schedule", "book", "cancel", "delete", "update", "read", "wake",
        "fetch", "list", "tell", "ask", "go",
    }

    def _decompose(message):
        parts = re.split(r',\s+and\s+|,\s+', message)
        parts = [re.sub(r'(?i)^and\s+', '', p).strip() for p in parts if p.strip()]
        if len(parts) == 1:
            candidates = [p.strip() for p in re.split(r'\s+and\s+', message, flags=re.IGNORECASE)]
            if (len(candidates) > 1
                    and all(len(p.split()) >= 3 for p in candidates)
                    and all(p.split()[0].lower().rstrip("'s") in _VERBS for p in candidates)):
                parts = candidates
        parts = [p for p in parts if len(p.split()) >= 2]
        return parts if len(parts) > 1 else [message]

    # ------------------------------------------------------------------ #
    # Main routing logic                                                   #
    # ------------------------------------------------------------------ #

    user_msgs = [m for m in messages if m["role"] == "user"]
    non_user  = [m for m in messages if m["role"] != "user"]

    def _resolve(content, depth=0, max_depth=2):
        """
        Recursively resolve a request on-device.
        1. Try fast-path + self-consistency on the content.
        2. If that fails and depth < max_depth, decompose and recurse.
        Returns (calls_list_or_None, time_ms_spent).
        """
        msgs = non_user + [{"role": "user", "content": content}]
        result, time_spent = _on_device(msgs, tools)
        if result is not None:
            return result["function_calls"], time_spent

        if depth < max_depth:
            parts = _decompose(content)
            if len(parts) > 1:
                all_calls = []
                total_time = time_spent
                for part in parts:
                    part_calls, part_time = _resolve(part, depth + 1, max_depth)
                    total_time += part_time
                    if part_calls is None:
                        return None, total_time
                    all_calls.extend(part_calls)
                return all_calls, total_time

        return None, time_spent

    # Step 1 + 2: fast-path then self-consistency on the full message.
    result, spent_time = _on_device(messages, tools)
    if result is not None:
        result["source"] = "on-device"
        return result

    # Step 3: recursive decomposition.
    if user_msgs:
        sub_contents = _decompose(user_msgs[-1]["content"])
        if len(sub_contents) > 1:
            all_calls = []
            total_time = spent_time
            success = True
            for sub_content in sub_contents:
                calls, sub_time = _resolve(sub_content)
                total_time += sub_time
                if calls is None:
                    success = False
                    break
                all_calls.extend(calls)

            if success:
                return {
                    "function_calls": all_calls,
                    "total_time_ms": total_time,
                    "source": "on-device",
                }

            cloud = generate_cloud(messages, tools)
            cloud["source"] = "cloud (fallback)"
            cloud["total_time_ms"] += total_time
            return cloud

    # Step 4: cloud fallback.
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += spent_time
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
