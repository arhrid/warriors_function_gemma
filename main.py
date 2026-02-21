import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time, random
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types
from collections import defaultdict, deque


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
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
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
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
        return {"function_calls": function_calls, "total_time_ms": total_time_ms}
    except Exception:
        return {"function_calls": [], "total_time_ms": 0}


def _coerce_args(arguments, tool):
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
    enhanced = []
    for t in tools:
        et = json.loads(json.dumps(t))
        props = et["parameters"].get("properties", {})
        hints = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", pname)
            hints.append(f"{pname}: {ptype} ({pdesc})")
        if hints:
            et["description"] = et["description"] + ". Parameters: " + ", ".join(hints)
        enhanced.append(et)
    return enhanced


def _validate_calls(function_calls, tools_by_name, tool_required):
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
    return [int(x) for x in re.findall(r'\b(\d+)\b', text)]


def _extract_time(text):
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
    props = tool["parameters"].get("properties", {})
    args = dict(call.get("arguments", {}))
    param_names = set(props.keys())
    if "hour" in param_names and "minute" in param_names:
        hour, minute = _extract_time(user_text)
        if hour is not None:
            args["hour"] = hour
            args["minute"] = minute
            return args
    for pname in param_names:
        if "minute" in pname and "hour" not in param_names:
            m = re.search(r'(\d+)\s*(?:minute|min)', user_text, re.IGNORECASE)
            if m:
                args[pname] = int(m.group(1))
                return args
    int_params = [k for k, v in props.items() if v.get("type") == "integer"]
    if len(int_params) == 1:
        numbers = _extract_numbers(user_text)
        if len(numbers) == 1:
            args[int_params[0]] = numbers[0]
    return args


def _fix_string_args(call, tool, user_text):
    props = tool["parameters"].get("properties", {})
    args = dict(call.get("arguments", {}))
    for key, pinfo in props.items():
        if key not in args:
            continue
        ptype = pinfo.get("type", "string")
        if ptype != "string":
            continue
        pdesc = (pinfo.get("description", "") + " " + key).lower()
        if "time" in pdesc:
            m = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.))', user_text, re.IGNORECASE)
            if not m:
                m = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.))', user_text)
            if m:
                args[key] = m.group(1)
    return args


def _try_construct_call(tool, user_text):
    """Construct a function call by extracting args from user text via regex."""
    props = tool["parameters"].get("properties", {})
    required = tool["parameters"].get("required", [])
    param_names = set(props.keys())
    args = {}

    for pname, pinfo in props.items():
        ptype = pinfo.get("type", "string")
        pdesc = (pinfo.get("description", "") + " " + pname).lower()

        if ptype == "integer":
            if "hour" in pname:
                h, _ = _extract_time(user_text)
                if h is not None:
                    args[pname] = h
            elif "minute" in pname and "hour" in param_names:
                _, m = _extract_time(user_text)
                if m is not None:
                    args[pname] = m
            elif "minute" in pname:
                # Duration minutes — require explicit "X minutes/min" context
                m = re.search(r'(\d+)\s*(?:minute|min)', user_text, re.IGNORECASE)
                if m:
                    args[pname] = int(m.group(1))
            else:
                nums = _extract_numbers(user_text)
                if nums:
                    args[pname] = nums[0]

        elif ptype == "string":
            val = None
            if any(w in pdesc for w in ("time", "when")):
                m = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.))', user_text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.))', user_text)
                if m:
                    val = m.group(1)
            elif any(w in pdesc for w in ("title", "subject", "reminder")):
                m = re.search(r'(?:remind(?:er)?\s+(?:me\s+)?(?:about|to|for)|about)\s+(.+?)(?:\s+at\s+\d|\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,!')
                    val = re.sub(r'^(?:the|a|an)\s+', '', val, flags=re.IGNORECASE)
            elif "recipient" in pname or (any(w in pdesc for w in ("person", "contact")) and "message" not in pname):
                m = re.search(r'(?:to)\s+(\w+)\s+(?:saying|say|that)', user_text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(?:message\s+to|send\s+to|text)\s+(\w+)', user_text, re.IGNORECASE)
                if not m:
                    names = re.findall(r'\b[A-Z][a-z]+\b', user_text)
                    skip = {"Send", "Set", "Get", "Create", "Remind", "Search",
                            "Find", "Look", "Play", "Text", "Check", "What",
                            "How", "The", "And", "Let", "Tell", "Shoot", "Ping"}
                    candidates = [n for n in names if n not in skip]
                    if candidates:
                        val = candidates[0]
                if m:
                    val = m.group(1)
            elif any(w in pdesc for w in ("message", "content", "text", "body")):
                m = re.search(r'(?:saying|say|that)\s+(.+?)(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,!')
            elif any(w in pdesc for w in ("song", "playlist", "music", "track")):
                # Keep "X music" when before "and" (e.g. "classical music and...")
                m = re.search(r'(?:play|listen to)\s+(?:some\s+)?(.+?\s+music)(?:,?\s+and\s+|,\s+)', user_text, re.IGNORECASE)
                if not m:
                    # Strip trailing "music" suffix at end of phrase
                    m = re.search(r'(?:play|listen to)\s+(?:some\s+)?(.+?)(?:\s+music)?(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,!')
            elif any(w in pdesc for w in ("query", "search")):
                m = re.search(r'(?:find|search for|look for|look up|search)\s+(\w+)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1)
            elif any(w in pdesc for w in ("location", "city", "place")):
                m = re.search(r'weather\s+(?:in|for)\s+(.+?)(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(?:in|for)\s+(.+?)(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,!')
            if val:
                args[pname] = val

    if all(r in args for r in required):
        return {"name": tool["name"], "arguments": args}
    return None


_ACTION_SYNONYMS = {
    "send": {"text"},
    "text": {"send", "message"},
    "search": {"find", "look"},
    "find": {"search", "look"},
    "look": {"search", "find"},
    "get": {"check"},
    "check": {"get"},
    "play": {"listen"},
    "listen": {"play"},
    "create": {"make"},
    "remind": {"reminder"},
    "reminder": {"remind"},
    "alarm": {"wake"},
    "wake": {"alarm"},
}


def _is_multi_request(text):
    lower = text.lower()
    if " and " not in lower and ", " not in lower:
        return False
    parts = re.split(r'\s+and\s+|,\s+', lower)
    if len(parts) < 2:
        return False
    action_words = {"set", "get", "send", "check", "find", "look", "play",
                    "remind", "text", "search", "create", "what", "how",
                    "tell", "let", "ping", "shoot"}
    action_parts = sum(1 for p in parts
                       if any(p.strip().startswith(w) for w in action_words))
    return action_parts >= 2


def _is_tool_relevant(tool, user_text):
    desc_words = set(re.findall(r'\w+', tool["description"].lower()))
    user_words = set(re.findall(r'\w+', user_text.lower()))
    name_words = set(tool["name"].lower().replace("_", " ").split())
    tool_words = desc_words | name_words
    for p in tool["parameters"].get("properties", {}).values():
        pdesc = p.get("description", "").lower()
        tool_words.update(re.findall(r'\w+', pdesc))
    expanded = set()
    for w in tool_words:
        if w in _ACTION_SYNONYMS:
            expanded.update(_ACTION_SYNONYMS[w])
    tool_words |= expanded
    overlap = tool_words & user_words
    stopwords = {"a", "an", "the", "for", "to", "of", "in", "is", "and", "or", "it"}
    meaningful_overlap = overlap - stopwords
    return len(meaningful_overlap) >= 1

# ── Query rewriter ────────────────────────────────────────────
REWRITE_RULES = [
    (r"wake me up at",              "set an alarm for"),
    (r"wake me up",                 "set an alarm for 7am"),
    (r"reach out to (\w+)",         r"send a message to \1 saying"),
    (r"what'?s?\s+it like in",      "get weather in"),
    (r"how'?s?\s+the weather in",   "get weather in"),
    (r"let (\w+) know",             r"send a message to \1 saying"),
    (r"shoot (\w+) a text",         r"send a message to \1 saying"),
    (r"ping (\w+)",                 r"send a message to \1 saying"),
    (r"tell (\w+) that",            r"send a message to \1 saying"),
    (r"play some",                  "play"),
    (r"what'?s?\s+the forecast in", "get weather in"),
]

def _rewrite_query(query: str) -> str:
    for pattern, replacement in REWRITE_RULES:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return query


# ── Intent counter ────────────────────────────────────────────
AND_KEYWORDS = [" and ", " also ", " plus ", " then ", " as well"]

def _count_intents(query: str) -> int:
    return 1 + sum(1 for kw in AND_KEYWORDS if kw in query.lower())


# ── Traffic shifter ───────────────────────────────────────────
class TrafficShifter:
    def __init__(
        self,
        window_size:        int   = 20,
        initial_local_prob: float = 0.80,
        min_local_prob:     float = 0.20,
        max_local_prob:     float = 0.99,
        shift_rate:         float = 0.08,
    ):
        self.min_local_prob = min_local_prob
        self.max_local_prob = max_local_prob
        self.shift_rate     = shift_rate
        self.outcomes       = defaultdict(lambda: deque(maxlen=window_size))
        self.local_prob     = defaultdict(lambda: initial_local_prob)

    def categorize(self, query: str) -> str:
        q            = query.lower()
        intent_count = _count_intents(query)
        if any(w in q for w in ["weather", "forecast"]):
            tool_type = "weather"
        elif any(w in q for w in ["alarm", "wake"]):
            tool_type = "alarm"
        elif any(w in q for w in ["message", "text", "tell", "ping", "send"]):
            tool_type = "message"
        elif any(w in q for w in ["timer", "countdown"]):
            tool_type = "timer"
        elif any(w in q for w in ["remind", "reminder"]):
            tool_type = "reminder"
        elif any(w in q for w in ["play", "music", "song"]):
            tool_type = "music"
        elif any(w in q for w in ["contact", "find", "look"]):
            tool_type = "contacts"
        else:
            tool_type = "other"
        return f"{intent_count}intent_{tool_type}"

    def should_use_local(self, category: str) -> bool:
        return random.random() < self.local_prob[category]

    def record_outcome(self, category: str, succeeded: bool):
        self.outcomes[category].append(succeeded)
        window = list(self.outcomes[category])
        if len(window) >= 3:
            success_rate = sum(window) / len(window)
            current      = self.local_prob[category]
            self.local_prob[category] = max(
                self.min_local_prob,
                min(
                    self.max_local_prob,
                    current + self.shift_rate * (success_rate - current)
                )
            )


# module level — persists across all calls
shifter = TrafficShifter()

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Smart hybrid: cactus selects tools, regex reconstructs args."""
    start = time.time()
    tools_by_name = {t["name"]: t for t in tools}
    tool_required = {t["name"]: t["parameters"].get("required", []) for t in tools}
    user_text = messages[-1]["content"] if messages else ""
    total_time = 0
    
    rewritten          = _rewrite_query(user_text)
    rewritten_messages = messages[:-1] + [
        {"role": "user", "content": rewritten}
    ]
    intent_count = _count_intents(user_text)
    n_tools = len(tools)
    
    category     = shifter.categorize(user_text)
    use_local    = shifter.should_use_local(category)

    # if shifter says cloud AND multi-intent → skip local entirely
    if not use_local and intent_count >= 2:
        cloud = generate_cloud(messages, tools)
        cloud["source"]        = "cloud (fallback)"
        cloud["total_time_ms"] = (time.time() - start) * 1000
        return cloud
    
    def _reconstruct_args(call):
        tool = tools_by_name.get(call["name"])
        if not tool:
            return
        constructed = _try_construct_call(tool, user_text)
        if constructed:
            call["arguments"] = constructed["arguments"]
        else:
            call["arguments"] = _fix_integer_args(call, tool, user_text)
            call["arguments"] = _fix_string_args(call, tool, user_text)

    # Phase 1: on-device with enhanced tools
    local = _run_cactus(rewritten_messages, tools, tool_rag_top_k=0, enhance=True)
    total_time += local["total_time_ms"]
    valid_calls = _validate_calls(local["function_calls"], tools_by_name, tool_required)

    # Filter out calls for tools not relevant to user text
    valid_calls = [c for c in valid_calls
                   if _is_tool_relevant(tools_by_name[c["name"]], user_text)]

    # Override args with regex extraction
    for call in valid_calls:
        _reconstruct_args(call)

    # Phase 1b: regex construction fallback
    if not valid_calls:
        for t in tools:
            if _is_tool_relevant(t, user_text):
                constructed = _try_construct_call(t, user_text)
                if constructed:
                    valid_calls.append(constructed)
                    if not _is_multi_request(user_text):
                        break

    # Phase 2: per-tool completion for multi-request
    if _is_multi_request(user_text):
        called_names = {c["name"] for c in valid_calls}
        for t in tools:
            if t["name"] in called_names:
                continue
            if not _is_tool_relevant(t, user_text):
                continue
            constructed = _try_construct_call(t, user_text)
            if constructed:
                valid_calls.append(constructed)
                called_names.add(t["name"])
                continue
            single = _run_cactus(messages, [t], tool_rag_top_k=0, enhance=True)
            total_time += single["total_time_ms"]
            extra = _validate_calls(single["function_calls"], tools_by_name, tool_required)
            for call in extra:
                if call["name"] not in called_names:
                    _reconstruct_args(call)
                    valid_calls.append(call)
                    called_names.add(call["name"])

    if valid_calls:
        shifter.record_outcome(category, succeeded=True)
        return {
            "function_calls": valid_calls,
            "total_time_ms": total_time,
            "confidence": local["confidence"],
            "source": "on-device",
        }

    # Phase 3: cloud fallback
    shifter.record_outcome(category, succeeded=False)
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += total_time
    return cloud


def print_result(label, result):
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


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]
    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]
    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)
    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)
    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
