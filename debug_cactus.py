"""Debug: see raw cactus output for ALL benchmark cases."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
from cactus import cactus_init, cactus_complete, cactus_destroy

model_path = os.path.join(os.path.dirname(__file__), "..", "cactus", "weights", "functiongemma-270m-it")

# Import benchmark cases
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent, "cactus", "python", "src"))

# Inline the failing cases with enhanced tool descriptions
from main import _enhance_tools, _coerce_args

CASES = [
    {
        "name": "alarm_10am",
        "messages": [{"role": "user", "content": "Set an alarm for 10 AM."}],
        "tools": [{"name": "set_alarm", "description": "Set an alarm for a given time",
                   "parameters": {"type": "object", "properties": {
                       "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                       "minute": {"type": "integer", "description": "Minute to set the alarm for"}},
                       "required": ["hour", "minute"]}}],
        "expected": [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}],
    },
    {
        "name": "reminder_meeting",
        "messages": [{"role": "user", "content": "Remind me about the meeting at 3:00 PM."}],
        "tools": [{"name": "create_reminder", "description": "Create a reminder with a title and time",
                   "parameters": {"type": "object", "properties": {
                       "title": {"type": "string", "description": "Reminder title"},
                       "time": {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"}},
                       "required": ["title", "time"]}}],
        "expected": [{"name": "create_reminder", "arguments": {"title": "meeting", "time": "3:00 PM"}}],
    },
    {
        "name": "reminder_among_four",
        "messages": [{"role": "user", "content": "Remind me to call the dentist at 2:00 PM."}],
        "tools": [
            {"name": "get_weather", "description": "Get current weather for a location",
             "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}},
            {"name": "send_message", "description": "Send a message to a contact",
             "parameters": {"type": "object", "properties": {"recipient": {"type": "string", "description": "Name"}, "message": {"type": "string", "description": "Message"}}, "required": ["recipient", "message"]}},
            {"name": "create_reminder", "description": "Create a reminder with a title and time",
             "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Reminder title"}, "time": {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"}}, "required": ["title", "time"]}},
            {"name": "set_alarm", "description": "Set an alarm for a given time",
             "parameters": {"type": "object", "properties": {"hour": {"type": "integer", "description": "Hour"}, "minute": {"type": "integer", "description": "Minute"}}, "required": ["hour", "minute"]}},
        ],
        "expected": [{"name": "create_reminder", "arguments": {"title": "call the dentist", "time": "2:00 PM"}}],
    },
    {
        "name": "message_and_weather",
        "messages": [{"role": "user", "content": "Send a message to Bob saying hi and get the weather in London."}],
        "tools": [
            {"name": "get_weather", "description": "Get current weather for a location",
             "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}},
            {"name": "send_message", "description": "Send a message to a contact",
             "parameters": {"type": "object", "properties": {"recipient": {"type": "string", "description": "Name"}, "message": {"type": "string", "description": "Message"}}, "required": ["recipient", "message"]}},
            {"name": "set_alarm", "description": "Set an alarm for a given time",
             "parameters": {"type": "object", "properties": {"hour": {"type": "integer", "description": "Hour"}, "minute": {"type": "integer", "description": "Minute"}}, "required": ["hour", "minute"]}},
        ],
        "expected": [
            {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
            {"name": "get_weather", "arguments": {"location": "London"}},
        ],
    },
]

PROMPTS = [
    ("enhanced", "You are a function calling assistant. Call the correct function(s) with accurate arguments extracted from the user request. Use exact values from the request. If multiple actions are needed, make ALL function calls."),
    ("baseline", "You are a helpful assistant that can use tools."),
]

for case in CASES:
    print(f"\n{'='*60}")
    print(f"CASE: {case['name']}")
    print(f"USER: {case['messages'][0]['content']}")
    print(f"EXPECTED: {json.dumps(case['expected'])}")

    for prompt_name, prompt in PROMPTS:
        for enhance in [True, False]:
            label = f"{prompt_name}+{'enhanced_tools' if enhance else 'raw_tools'}"
            model = cactus_init(model_path)
            actual_tools = _enhance_tools(case["tools"]) if enhance else case["tools"]
            cactus_tools = [{"type": "function", "function": t} for t in actual_tools]
            raw_str = cactus_complete(
                model,
                [{"role": "system", "content": prompt}] + case["messages"],
                tools=cactus_tools,
                force_tools=True,
                max_tokens=512,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
                tool_rag_top_k=0,
                confidence_threshold=0.05,
            )
            cactus_destroy(model)
            try:
                raw = json.loads(raw_str)
                calls = raw.get("function_calls", [])
                conf = raw.get("confidence", 0)
                print(f"  [{label}] conf={conf:.4f} calls={json.dumps(calls)}")
            except:
                print(f"  [{label}] PARSE ERROR: {raw_str[:200]}")
