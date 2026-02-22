import sys, os

# Resolve paths relative to this file so they work from any CWD
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, "cactus", "python", "src"))
functiongemma_path = os.path.join(_this_dir, "cactus", "weights", "functiongemma-270m-it")

import json, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

# Global model cache to avoid repeated init/destroy overhead
_cached_model = None


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


_cloud_client = None


def _get_cloud_client():
    global _cloud_client
    if _cloud_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            _cloud_client = genai.Client(api_key=api_key)
        except Exception:
            return None
    return _cloud_client


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = _get_cloud_client()
    if not client:
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
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=(
                    "You are a function calling assistant. "
                    "Call the correct function(s) with accurate arguments. "
                    "Use exact values from the user request."
                ),
            ),
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
        elif expected_type == "number" and isinstance(val, int):
            coerced[key] = float(val)
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
    seen_calls = set()
    for fc in function_calls:
        name = fc.get("name", "")
        args = fc.get("arguments", {})
        if name in tools_by_name:
            required = tool_required.get(name, [])
            if all(r in args for r in required):
                coerced_args = _coerce_args(args, tools_by_name[name])
                call_sig = (name, tuple(sorted(
                    (k, str(v)) for k, v in coerced_args.items())))
                if call_sig not in seen_calls:
                    valid.append({"name": name, "arguments": coerced_args})
                    seen_calls.add(call_sig)
    return valid


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = cactus_init(functiongemma_path)
    return _cached_model


def _run_cactus(messages, tools, system_prompt=None, tool_rag_top_k=0, enhance=True):
    model = _get_model()
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
    # 12-hour format with AM/PM
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
    # 24-hour format: "14:30", "09:10" (not inside a date like 2026-02-22)
    match = re.search(r'(?:^|[\s,at])(\d{1,2}):(\d{2})(?:\s|$|[,;.])', text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
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


def _strip_quotes(val):
    """Strip surrounding quotes from a value."""
    if val and len(val) >= 2 and val[0] in ('"', "'", '\u201c') and val[-1] in ('"', "'", '\u201d'):
        return val[1:-1]
    return val


def _try_construct_call(tool, user_text):
    """Construct a function call by extracting args from user text via regex.
    Returns None for tools with unfamiliar parameter patterns (lets cactus handle them).
    """
    props = tool["parameters"].get("properties", {})
    required = tool["parameters"].get("required", [])
    param_names = set(props.keys())
    args = {}
    has_unknown_params = False
    # Temporal words to strip from location extraction
    temporal_stop = r'(?:\s+(?:right now|today|tomorrow|tonight|this|currently|at the moment))'

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
                m = re.search(r'(\d+)\s*(?:minute|min)', user_text, re.IGNORECASE)
                if m:
                    args[pname] = int(m.group(1))
            else:
                nums = _extract_numbers(user_text)
                if nums:
                    args[pname] = nums[0]

        elif ptype == "string":
            val = None
            matched_pattern = False
            if any(w in pdesc for w in ("time", "when")):
                matched_pattern = True
                m = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.))', user_text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.))', user_text)
                if m:
                    val = m.group(1)
            elif any(w in pdesc for w in ("title", "subject", "reminder")):
                matched_pattern = True
                # "remind me to X at TIME" or "reminder to X"
                m = re.search(r'(?:remind(?:er)?\s+(?:me\s+)?(?:about|to|for)|(?:reminder\s+to))\s+(.+?)(?:\s+at\s+\d|\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,!')
                    val = re.sub(r'^(?:the|a|an)\s+', '', val, flags=re.IGNORECASE)
            elif "recipient" in pname or (any(w in pdesc for w in ("person", "contact")) and "message" not in pname):
                matched_pattern = True
                # "to NAME saying/:" or "message to NAME" or "text NAME"
                m = re.search(r'(?:to)\s+(\w+)\s+(?:saying|say|that|:)', user_text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(?:message\s+to|send\s+to|text)\s+(\w+)', user_text, re.IGNORECASE)
                if not m:
                    # "Send NAME a/:" or "Ping NAME:" pattern
                    m = re.search(r'(?:send|ping|text)\s+([A-Z][a-z]+)\s*(?:a\s+|:)', user_text)
                if not m:
                    names = re.findall(r'\b[A-Z][a-z]+\b', user_text)
                    skip = {"Send", "Set", "Get", "Create", "Remind", "Search",
                            "Find", "Look", "Play", "Text", "Check", "What",
                            "How", "The", "And", "Let", "Tell", "Shoot", "Ping",
                            "Book", "Add", "Convert", "Translate", "Schedule",
                            "Enable", "Turn", "Start", "Compare", "Put", "Do"}
                    candidates = [n for n in names if n not in skip]
                    if candidates:
                        val = candidates[0]
                if m:
                    val = m.group(1)
            elif any(w in pdesc for w in ("message", "content", "text", "body")):
                matched_pattern = True
                # Quoted content: 'see you at 8' or "joining in 5"
                m = re.search(r"""['"\u201c](.+?)['"\u201d]""", user_text)
                if not m:
                    # "saying X" or ": X" separator patterns
                    m = re.search(r'(?:saying|say|that)\s+(.+?)(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if not m:
                    # "NAME: message" or "note: message" pattern
                    m = re.search(r'(?:[A-Z][a-z]+|note|message)\s*:\s*(.+?)(?:\s*$)', user_text, re.IGNORECASE)
                if not m:
                    # "send: message" pattern
                    m = re.search(r'(?:send)\s*:\s*(.+?)(?:\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,!')
            elif any(w in pdesc for w in ("song", "playlist", "music", "track")):
                matched_pattern = True
                m = re.search(r'(?:play|listen to)\s+(?:some\s+)?(.+?\s+music)(?:,?\s+and\s+|,\s+)', user_text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(?:play|listen to|put on)\s+(?:some\s+)?(.+?)(?:\s+music)?(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = _strip_quotes(m.group(1).strip().rstrip('.,!'))
            elif any(w in pdesc for w in ("query", "search")):
                matched_pattern = True
                m = re.search(r'(?:find|search for|look for|look up|search)\s+(\w+)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1)
            elif any(w in pdesc for w in ("location", "city", "place")):
                matched_pattern = True
                m = re.search(r'(?:weather|forecast)\s+(?:in|for)\s+(.+?)' + temporal_stop + r'?(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if not m:
                    m = re.search(r'(?:in|for)\s+(.+?)' + temporal_stop + r'?(?:\s+and\s+|,\s*|\s*[.!?]?\s*$)', user_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,!')
                    val = re.sub(r'\s+(?:right now|today|tomorrow|tonight)$', '', val, flags=re.IGNORECASE)
            elif any(w in pdesc for w in ("date", "day", "when")):
                matched_pattern = True
                m = re.search(r'(\d{4}-\d{2}-\d{2})', user_text)
                if m:
                    val = m.group(1)
                elif "tomorrow" in user_text.lower():
                    from datetime import date, timedelta
                    val = (date.today() + timedelta(days=1)).isoformat()
                elif "today" in user_text.lower():
                    from datetime import date
                    val = date.today().isoformat()
            elif any(w in pdesc for w in ("start_time", "end_time", "start time", "end time", "datetime")):
                matched_pattern = True
                dts = re.findall(r'(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})', user_text)
                if dts:
                    if "start" in pname and dts:
                        val = f"{dts[0][0]} {dts[0][1]}"
                    elif "end" in pname and len(dts) >= 2:
                        val = f"{dts[1][0]} {dts[1][1]}"
            elif any(w in pdesc for w in ("language", "target_language", "source_language")):
                matched_pattern = True
                langs = ["spanish", "french", "german", "italian", "portuguese",
                         "japanese", "chinese", "korean", "arabic", "hindi",
                         "russian", "english", "dutch", "swedish"]
                for lang in langs:
                    if lang in user_text.lower():
                        val = lang.capitalize()
                        break
            else:
                # Check for enum hints in description
                enum_match = re.search(r'(?:One of|one of|Possible values?|Options?):\s*(.+?)(?:\.|$)', pinfo.get("description", ""))
                if enum_match:
                    matched_pattern = True
                    choices = [c.strip().strip("'\"") for c in re.split(r'[,;]|\bor\b', enum_match.group(1))]
                    user_lower = user_text.lower()
                    for choice in choices:
                        if choice.lower() in user_lower or choice.replace("_", " ").lower() in user_lower:
                            val = choice
                            break
                else:
                    has_unknown_params = True
            if val:
                val = _strip_quotes(val)
                args[pname] = val

        elif ptype == "array":
            item_type = pinfo.get("items", {}).get("type", "string")
            if item_type == "string":
                # Try quoted items first
                items = re.findall(r"'([^']+)'", user_text)
                if not items:
                    items = re.findall(r'"([^"]+)"', user_text)
                if not items:
                    # "add X, Y, and Z to list/cart"
                    if any(w in pdesc for w in ("item", "list", "add", "shop", "grocer")):
                        m = re.search(r'(?:add|put|get|buy)\s+(.+?)(?:\s+to\s+|\s+on\s+|\s+from\s+)', user_text, re.IGNORECASE)
                        if m:
                            raw = m.group(1)
                            items = re.split(r',\s*(?:and\s+)?|\s+and\s+', raw)
                            items = [i.strip() for i in items if i.strip()]
                    elif any(w in pdesc for w in ("attendee", "invite", "email")):
                        items = re.findall(r'[\w.+-]+@[\w.-]+\.\w+', user_text)
                    elif any(w in pdesc for w in ("contact", "name", "person", "call", "bypass")):
                        m = re.search(r'(?:from|allow|let)\s+(.+?)(?:\s+through|\s+bypass|$)', user_text, re.IGNORECASE)
                        if m:
                            raw = m.group(1)
                            items = re.split(r',\s*(?:and\s+)?|\s+and\s+', raw)
                            items = [i.strip() for i in items if i.strip()]
                if items:
                    args[pname] = items
                else:
                    has_unknown_params = True
            else:
                has_unknown_params = True

        elif ptype == "object":
            has_unknown_params = True

        elif ptype == "number":
            nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', user_text)
            if nums:
                args[pname] = float(nums[0]) if '.' in nums[0] else int(nums[0])

    if has_unknown_params and not all(r in args for r in required):
        return None
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
    "play": {"listen", "put"},
    "listen": {"play"},
    "put": {"play"},
    "create": {"make", "schedule", "add"},
    "schedule": {"create", "make"},
    "remind": {"reminder"},
    "reminder": {"remind"},
    "alarm": {"wake"},
    "wake": {"alarm"},
    "convert": {"exchange"},
    "exchange": {"convert"},
    "translate": {"translation"},
    "book": {"reserve", "ride"},
    "forecast": {"weather", "outlook"},
    "ping": {"send", "message", "text"},
    "note": {"message", "send"},
}


def _is_multi_request(text):
    lower = text.lower()
    action_words = {"set", "get", "send", "check", "find", "look", "play",
                    "remind", "text", "search", "create", "what", "how",
                    "tell", "let", "ping", "shoot", "book", "convert",
                    "translate", "add", "enable", "schedule", "compare",
                    "turn", "put", "start", "message", "also"}
    # Check conjunctions
    if " and " in lower or ", " in lower or "; " in lower or " then " in lower:
        parts = re.split(r',\s+and\s+|\s+and\s+|,\s+|\s*;\s*|\s+then\s+', lower)
        if len(parts) >= 2:
            action_parts = sum(1 for p in parts
                               if any(p.strip().startswith(w) for w in action_words))
            if action_parts >= 2:
                return True
    # Check period-separated sentences: "Set alarm. Check weather."
    if ". " in lower:
        sentences = [s.strip() for s in lower.split(". ") if s.strip()]
        if len(sentences) >= 2:
            action_parts = sum(1 for s in sentences
                               if any(s.startswith(w) for w in action_words))
            if action_parts >= 2:
                return True
    return False


def _relevance_score(tool, user_text):
    """Score how relevant a tool is to the user text (higher = more relevant)."""
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
    stopwords = {"a", "an", "the", "for", "to", "of", "in", "is", "and", "or",
                 "it", "at", "on", "by", "with", "from", "one", "e", "g"}
    meaningful_overlap = overlap - stopwords
    # Bonus for matching tool name words (stronger signal)
    name_overlap = name_words & user_words
    return len(meaningful_overlap) + len(name_overlap)


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
    stopwords = {"a", "an", "the", "for", "to", "of", "in", "is", "and", "or",
                 "it", "at", "on", "by", "with", "from", "one", "e", "g"}
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
    (r"ping (\w+)\s*:",             r"send a message to \1 saying"),
    (r"ping (\w+)",                 r"send a message to \1 saying"),
    (r"tell (\w+) that",            r"send a message to \1 saying"),
    (r"play some",                  "play"),
    (r"what'?s?\s+the forecast in", "get weather in"),
    (r"do i need (?:an? )?(?:umbrella|jacket|coat) in", "get weather in"),
    (r"put on\s+(?:some\s+)?",      "play "),
    (r"start a (\d+)\s*(?:minute|min)\s*(?:countdown|timer)", r"set a timer for \1 minutes"),
    (r"(\d+)\s*(?:minute|min)\s*countdown", r"\1 minute timer"),
    (r"set a reminder\b",          "remind me"),
    (r"what'?s?\s+the weather\b",   "get weather"),
    (r"can you (?:please )?",       ""),
    (r"could you (?:please )?",     ""),
    (r"i need (?:you )?to ",        ""),
    (r"i want (?:you )?to ",        ""),
    (r"grab (?:an? )?(?:ride|cab|uber|lyft)", "book a ride"),
    (r"order (?:an? )?(?:ride|cab|uber|lyft)", "book a ride"),
    (r"get me (?:an? )?(?:ride|cab)", "book a ride"),
]


def _rewrite_query(query):
    for pattern, replacement in REWRITE_RULES:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return query


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Smart hybrid: regex-first for known patterns, cactus on-device fallback,
    cloud only as last resort. Maximizes on-device ratio and speed."""
    tools_by_name = {t["name"]: t for t in tools}
    tool_required = {t["name"]: t["parameters"].get("required", []) for t in tools}
    user_text = messages[-1]["content"] if messages else ""
    total_time = 0

    rewritten = _rewrite_query(user_text)
    rewritten_messages = messages[:-1] + [{"role": "user", "content": rewritten}]
    # Use rewritten text for ALL matching & extraction (regex + relevance),
    # not just cactus. This ensures rewrites like "ping X:" → "send a message
    # to X saying" benefit the regex path too.
    user_text = rewritten
    is_multi = _is_multi_request(user_text)

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

    # Phase 0: regex-first for single-tool cases (instant, no model needed)
    relevant_tools = [t for t in tools if _is_tool_relevant(t, user_text)]
    best_tool = None
    if not is_multi and len(relevant_tools) >= 1:
        # Pick the tool with highest relevance score
        scored = []
        for t in relevant_tools:
            score = _relevance_score(t, user_text)
            scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        # Use regex if top tool is clearly dominant (2x runner-up)
        if len(scored) == 1 or scored[0][0] >= 2 * scored[1][0]:
            best_tool = scored[0][1]
    if best_tool and not is_multi:
        constructed = _try_construct_call(best_tool, user_text)
        if constructed:
            return {
                "function_calls": [constructed],
                "total_time_ms": 0,
                "confidence": 1.0,
                "source": "on-device",
            }

    # Phase 0b: regex-first for multi-tool requests
    if is_multi and len(relevant_tools) >= 2:
        regex_calls = []
        for t in relevant_tools:
            constructed = _try_construct_call(t, user_text)
            if constructed:
                regex_calls.append(constructed)
        if len(regex_calls) >= 2:
            return {
                "function_calls": regex_calls,
                "total_time_ms": 0,
                "confidence": 1.0,
                "source": "on-device",
            }

    # Phase 1: on-device cactus with enhanced tool descriptions
    local = _run_cactus(rewritten_messages, tools, tool_rag_top_k=0, enhance=True)
    total_time += local["total_time_ms"]
    valid_calls = _validate_calls(local["function_calls"], tools_by_name, tool_required)

    # Filter out calls for tools not relevant to user text
    valid_calls = [c for c in valid_calls
                   if _is_tool_relevant(tools_by_name[c["name"]], user_text)]

    # Override args with regex extraction where possible
    for call in valid_calls:
        _reconstruct_args(call)

    # Phase 1b: regex construction fallback
    if not valid_calls:
        for t in tools:
            if _is_tool_relevant(t, user_text):
                constructed = _try_construct_call(t, user_text)
                if constructed:
                    valid_calls.append(constructed)
                    if not is_multi:
                        break

    # Phase 2: per-tool completion for multi-request
    if is_multi:
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
            single = _run_cactus(rewritten_messages, [t], tool_rag_top_k=0, enhance=True)
            total_time += single["total_time_ms"]
            extra = _validate_calls(single["function_calls"], tools_by_name, tool_required)
            for call in extra:
                if call["name"] not in called_names:
                    _reconstruct_args(call)
                    valid_calls.append(call)
                    called_names.add(call["name"])

    if valid_calls:
        return {
            "function_calls": valid_calls,
            "total_time_ms": total_time,
            "confidence": local["confidence"],
            "source": "on-device",
        }

    # Phase 3: cloud fallback — only when local completely fails
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
