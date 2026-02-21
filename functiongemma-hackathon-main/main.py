
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, random, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types
from collections import defaultdict, deque


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
        model="gemini-2.0-flash",
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

SEND_PATTERNS = [
    # "send a message to NAME saying CONTENT"
    (r"(?:send(?:\s+a)?\s+message\s+to|text|tell|ping)\s+"
     r"(\w+)\s+saying\s+(.+?)(?:\.|$|,\s*and\b)",          1.0),
    # "let NAME know CONTENT"
    (r"let\s+(\w+)\s+know\s+(?:that\s+)?(.+?)(?:\.|$|,\s*and\b)", 0.9),
    # "reach out to NAME saying CONTENT"
    (r"reach\s+out\s+to\s+(\w+)\s+saying\s+(.+?)(?:\.|$)",  0.9),
    # "shoot NAME a text saying CONTENT"
    (r"shoot\s+(\w+)\s+a\s+text\s+saying\s+(.+?)(?:\.|$)",  0.9),
    # "text NAME saying CONTENT"
    (r"(?:text|message)\s+(\w+)\s+saying\s+(.+?)(?:\.|$)",  0.85),
]

# Node 1 - symbolic extracter 
def symbolic_extract(query: str) -> dict:
    """
    Pure regex extraction for send_message user input.
    Returns recipient, message, confidence.
    Runs in ~1ms on edge.
    """
    tool_names_in_query = query.lower()
    if not any(w in tool_names_in_query for w in
               ["message", "text", "tell", "ping",
                "send", "reach out", "shoot", "let"]):
        return {"recipient": None, "message": None, "confidence": 0.0}

    for pattern, confidence in SEND_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            recipient = match.group(1).strip().title()
            message   = match.group(2).strip().strip("'\"") \
                        if match.lastindex >= 2 else None
            if message:
                message = re.sub(
                    r"^(saying|that|:)\s+", "",
                    message, flags=re.IGNORECASE
                ).strip()
            return {
                "recipient":  recipient,
                "message":    message,
                "confidence": confidence if message else 0.0
            }

    return {"recipient": None, "message": None, "confidence": 0.0}

#Node 2 - Query rewriter (make indirect queries direct before 
# hitting FunctionGemma) - support edge 
REWRITE_RULES = [
    (r"wake me up at",              "set an alarm for"),
    (r"wake me up",                 "set an alarm for 7am"),
    (r"reach out to (\w+)",         r"send a message to \1 saying"),
    (r"what'?s?\s+it like in",      "get weather in"),
    (r"how'?s?\s+the weather in",   "get weather in"),
    (r"let (\w+) know",             r"send a message to \1 saying"),
    (r"look up (\w+)",              r"search contacts for \1"),
    (r"find (\w+) in.*contacts",    r"search contacts for \1"),
    (r"text (\w+) saying",          r"send a message to \1 saying"),
    (r"shoot (\w+) a text",         r"send a message to \1 saying"),
    (r"ping (\w+)",                 r"send a message to \1 saying"),
    (r"tell (\w+) that",            r"send a message to \1 saying"),
    (r"play some",                  "play"),
    (r"count down (\d+)",           r"set a timer for \1"),
    (r"what'?s?\s+the forecast in", "get weather in"),
]

def rewrite_query(query: str) -> str:
    for pattern, replacement in REWRITE_RULES:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return query

#Node 3 - Tool Pruner (reduces tool calls for FunctionGemma, fewer tools
# may reduce tool call incorrectness - improve F1 score)

TOOL_KEYWORDS = {
    "get_weather":     ["weather", "temperature", "forecast",
                        "like in", "outside", "rain"],
    "set_alarm":       ["alarm", "wake", "wake up", "morning"],
    "set_timer":       ["timer", "countdown", "count down", "minutes"],
    "send_message":    ["message", "text", "tell", "send",
                        "ping", "reach out"],
    "create_reminder": ["remind", "reminder", "don't forget", "remember"],
    "search_contacts": ["find", "look up", "search", "contact"],
    "play_music":      ["play", "music", "song", "playlist",
                        "beats", "jazz", "classical"],
}

def prune_tools(query: str, tools: list) -> list:
    query_lower = query.lower()
    scored = [(
        sum(1 for kw in TOOL_KEYWORDS.get(t["name"], [])
            if kw in query_lower),
        t
    ) for t in tools]
    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [t for s, t in scored if s > 0]
    # always keep at least 2 tools to avoid over-pruning
    return relevant if len(relevant) >= 2 else [t for _, t in scored[:2]]

# Node 4 - Intent Counter & Router 
# decide if local || try_local || cloud before model call 


AND_KEYWORDS = [" and ", " also ", " plus ", " then ", " as well"]

def count_intents(query: str) -> int:
    q = query.lower()
    return 1 + sum(1 for kw in AND_KEYWORDS if kw in q)

#Node 5 - Validator (Checks FunctionGemma output before trusting it)

def is_valid(result: dict, tools: list, expected_intents: int) -> bool:
    calls = result.get("function_calls", [])
    if not calls:
        return False
    valid_names = {t["name"] for t in tools}
    for call in calls:
        if call["name"] not in valid_names:
            return False
        tool     = next(t for t in tools if t["name"] == call["name"])
        required = tool["parameters"].get("required", [])
        if not all(r in call.get("arguments", {}) for r in required):
            return False
    # multi-intent: need at least as many calls as intents
    if expected_intents >= 2 and len(calls) < expected_intents:
        return False
    return True

#Node 6 - Traffic Shifter (shifts query local/cloud , learns from outcomes
# in real time)

class TrafficShifter:
    def __init__(
        self,
        window_size:        int   = 20,
        initial_local_prob: float = 0.65,
        min_local_prob:     float = 0.10,
        max_local_prob:     float = 0.95,
        shift_rate:         float = 0.08,
    ):
        self.window_size        = window_size
        self.min_local_prob     = min_local_prob
        self.max_local_prob     = max_local_prob
        self.shift_rate         = shift_rate
        self.outcomes:    dict   = defaultdict(lambda: deque(maxlen=window_size))
        self.local_prob:  dict   = defaultdict(lambda: initial_local_prob)

    def categorize(self, query: str, tools: list) -> str:
        """Map query → category. Similar queries share same traffic ratio."""
        q            = query.lower()
        intent_count = count_intents(query)

        if any(w in q for w in ["weather", "forecast", "like in"]):
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
        elif any(w in q for w in ["contact", "find", "look up"]):
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
            success_rate   = sum(window) / len(window)
            current        = self.local_prob[category]
            self.local_prob[category] = max(
                self.min_local_prob,
                min(
                    self.max_local_prob,
                    current + self.shift_rate * (success_rate - current)
                )
            )


# Module-level — persists across all generate_hybrid calls
shifter = TrafficShifter()



def generate_hybrid(messages, tools, confidence_threshold=0.99):
    start = time.time()
    query = messages[-1]["content"]
    intent_count = count_intents(query)
    tool_names = [t["name"] for t in tools]
    
    # Symbolic extraction
     # Only for send_message, single intent queries
    if "send_message" in tool_names and intent_count == 1:
        features = symbolic_extract(query)
        if features["confidence"] >= confidence_threshold:
            # Perfect extraction — skip ALL model calls
            return {
                "function_calls": [{
                    "name": "send_message",
                    "arguments": {
                        "recipient": features["recipient"],
                        "message":   features["message"],
                    }
                }],
                "source":        "on-device",
                "total_time_ms": (time.time() - start) * 1000,
            }
    #Rewrite query
    rewritten          = rewrite_query(query)
    rewritten_messages = messages[:-1] + [
        {"role": "user", "content": rewritten}
    ]
    
    # Prune tools
    pruned_tools = prune_tools(rewritten, tools)

    # Router 
    # Hard override: 3+ intents always cloud 
    # FunctionGemma cannot reliably handle 3 simultaneous tools
    if intent_count >= 3:
        cloud = generate_cloud(messages, tools)
        cloud["source"]        = "cloud (fallback)"
        cloud["total_time_ms"] = (time.time() - start) * 1000
        return cloud
    
    # Progressive traffic decision
    category  = shifter.categorize(query, tools)
    use_local = shifter.should_use_local(category)

    if use_local:
        # ── FunctionGemma (edge model) ────────────────────────
        local       = generate_cactus(rewritten_messages, pruned_tools)
        local_conf  = local.get("confidence", 0)

        # ── NODE 6: Validate output (edge, ~0ms) ─────────────
        local_valid = is_valid(local, tools, intent_count)

        if local_valid and local_conf > confidence_threshold:
            # Record success → shifter increases local probability
            shifter.record_outcome(category, succeeded=True)
            local["source"]        = "on-device"
            local["total_time_ms"] = (time.time() - start) * 1000
            return local

        # Local failed → record failure → shifter decreases local probability
        shifter.record_outcome(category, succeeded=False)
        cloud = generate_cloud(messages, tools)
        cloud["source"]        = "cloud (fallback)"
        cloud["total_time_ms"] = (time.time() - start) * 1000
        return cloud

    else:
        # Traffic shifted to cloud for this category
        cloud = generate_cloud(messages, tools)
        cloud["source"]        = "cloud (fallback)"
        cloud["total_time_ms"] = (time.time() - start) * 1000
        return cloud
    
    # """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
    # local = generate_cactus(messages, tools)

    # if local["confidence"] >= confidence_threshold:
    #     local["source"] = "on-device"
    #     return local

    # cloud = generate_cloud(messages, tools)
    # cloud["source"] = "cloud (fallback)"
    # cloud["local_confidence"] = local["confidence"]
    # cloud["total_time_ms"] += local["total_time_ms"]
    # return cloud


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
