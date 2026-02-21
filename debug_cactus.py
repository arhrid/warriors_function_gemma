"""Debug script: see raw cactus output for failing cases."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset

model_path = os.path.join(os.path.dirname(__file__), "..", "cactus", "weights", "functiongemma-270m-it")
model = cactus_init(model_path)

FAILING_CASES = [
    {
        "name": "alarm_10am",
        "messages": [{"role": "user", "content": "Set an alarm for 10 AM."}],
        "tools": [{"name": "set_alarm", "description": "Set an alarm for a given time",
                   "parameters": {"type": "object", "properties": {
                       "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                       "minute": {"type": "integer", "description": "Minute to set the alarm for"}},
                       "required": ["hour", "minute"]}}],
    },
    {
        "name": "timer_5min",
        "messages": [{"role": "user", "content": "Set a timer for 5 minutes."}],
        "tools": [{"name": "set_timer", "description": "Set a countdown timer",
                   "parameters": {"type": "object", "properties": {
                       "minutes": {"type": "integer", "description": "Number of minutes"}},
                       "required": ["minutes"]}}],
    },
    {
        "name": "reminder_meeting",
        "messages": [{"role": "user", "content": "Remind me about the meeting at 3:00 PM."}],
        "tools": [{"name": "create_reminder", "description": "Create a reminder with a title and time",
                   "parameters": {"type": "object", "properties": {
                       "title": {"type": "string", "description": "Reminder title"},
                       "time": {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"}},
                       "required": ["title", "time"]}}],
    },
    {
        "name": "message_alice",
        "messages": [{"role": "user", "content": "Send a message to Alice saying good morning."}],
        "tools": [{"name": "send_message", "description": "Send a message to a contact",
                   "parameters": {"type": "object", "properties": {
                       "recipient": {"type": "string", "description": "Name of the person to send the message to"},
                       "message": {"type": "string", "description": "The message content to send"}},
                       "required": ["recipient", "message"]}}],
    },
    {
        "name": "music_among_three",
        "messages": [{"role": "user", "content": "Play some jazz music."}],
        "tools": [
            {"name": "set_alarm", "description": "Set an alarm for a given time",
             "parameters": {"type": "object", "properties": {
                 "hour": {"type": "integer", "description": "Hour"}, "minute": {"type": "integer", "description": "Minute"}},
                 "required": ["hour", "minute"]}},
            {"name": "play_music", "description": "Play a song or playlist",
             "parameters": {"type": "object", "properties": {
                 "song": {"type": "string", "description": "Song or playlist name"}},
                 "required": ["song"]}},
            {"name": "get_weather", "description": "Get current weather for a location",
             "parameters": {"type": "object", "properties": {
                 "location": {"type": "string", "description": "City name"}},
                 "required": ["location"]}},
        ],
    },
]

for case in FAILING_CASES:
    print(f"\n{'='*60}")
    print(f"CASE: {case['name']}")
    print(f"USER: {case['messages'][0]['content']}")
    print(f"TOOLS: {[t['name'] for t in case['tools']]}")

    cactus_reset(model)
    cactus_tools = [{"type": "function", "function": t} for t in case["tools"]]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a function calling assistant. Analyze the user request and call the correct function(s) with accurate arguments."}] + case["messages"],
        tools=cactus_tools,
        force_tools=True,
        max_tokens=512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=0,
        confidence_threshold=0.05,
    )

    try:
        raw = json.loads(raw_str)
        print(f"CONFIDENCE: {raw.get('confidence', 'N/A')}")
        print(f"CLOUD_HANDOFF: {raw.get('cloud_handoff', 'N/A')}")
        print(f"FUNCTION_CALLS: {json.dumps(raw.get('function_calls', []), indent=2)}")
        print(f"RESPONSE: {raw.get('response', 'N/A')}")
    except json.JSONDecodeError:
        print(f"RAW (unparseable): {raw_str[:500]}")

cactus_destroy(model)
