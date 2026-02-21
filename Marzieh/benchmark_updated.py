import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json
from main import generate_hybrid


############## Tool definitions ##############

TOOL_GET_WEATHER = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"],
    },
}

TOOL_SET_ALARM = {
    "name": "set_alarm",
    "description": "Set an alarm for a given time",
    "parameters": {
        "type": "object",
        "properties": {
            "hour": {"type": "integer", "description": "Hour to set the alarm for"},
            "minute": {"type": "integer", "description": "Minute to set the alarm for"},
        },
        "required": ["hour", "minute"],
    },
}

TOOL_SEND_MESSAGE = {
    "name": "send_message",
    "description": "Send a message to a contact",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Name of the person to send the message to"},
            "message": {"type": "string", "description": "The message content to send"},
        },
        "required": ["recipient", "message"],
    },
}

TOOL_CREATE_REMINDER = {
    "name": "create_reminder",
    "description": "Create a reminder with a title and time",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "time": {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"},
        },
        "required": ["title", "time"],
    },
}

TOOL_SEARCH_CONTACTS = {
    "name": "search_contacts",
    "description": "Search for a contact by name",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Name to search for"},
        },
        "required": ["query"],
    },
}

TOOL_PLAY_MUSIC = {
    "name": "play_music",
    "description": "Play a song or playlist",
    "parameters": {
        "type": "object",
        "properties": {
            "song": {"type": "string", "description": "Song or playlist name"},
        },
        "required": ["song"],
    },
}

TOOL_SET_TIMER = {
    "name": "set_timer",
    "description": "Set a countdown timer",
    "parameters": {
        "type": "object",
        "properties": {
            "minutes": {"type": "integer", "description": "Number of minutes"},
        },
        "required": ["minutes"],
    },
}


############## Benchmark cases (NEW) ##############
# Notes:
# - Same tool schema + same scoring logic.
# - Completely new utterances, names, places, and compositions.
# - Keeps the same easy/medium/hard mix (10/10/10) to be comparable.

BENCHMARKS = [
    # ===== Easy: 1 tool, direct request =====
    {
        "name": "weather_vancouver",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "What's the weather in Vancouver right now?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "Vancouver"}}],
    },
    {
        "name": "alarm_645am",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Please set an alarm for 6:45 AM."}],
        "tools": [TOOL_SET_ALARM],
        "expected_calls": [{"name": "set_alarm", "arguments": {"hour": 6, "minute": 45}}],
    },
    {
        "name": "message_priya_checking_in",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Text Priya: checking in—are we still on for today?"}],
        "tools": [TOOL_SEND_MESSAGE],
        "expected_calls": [
            {"name": "send_message", "arguments": {"recipient": "Priya", "message": "checking in—are we still on for today?"}}
        ],
    },
    {
        "name": "timer_12min",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Set a timer for 12 minutes."}],
        "tools": [TOOL_SET_TIMER],
        "expected_calls": [{"name": "set_timer", "arguments": {"minutes": 12}}],
    },
    {
        "name": "reminder_pay_rent",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Remind me to pay rent at 9:30 AM."}],
        "tools": [TOOL_CREATE_REMINDER],
        "expected_calls": [{"name": "create_reminder", "arguments": {"title": "pay rent", "time": "9:30 AM"}}],
    },
    {
        "name": "search_contact_omar",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Find Omar in my contacts."}],
        "tools": [TOOL_SEARCH_CONTACTS],
        "expected_calls": [{"name": "search_contacts", "arguments": {"query": "Omar"}}],
    },
    {
        "name": "play_take_five",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Play \"Take Five\"."}],
        "tools": [TOOL_PLAY_MUSIC],
        "expected_calls": [{"name": "play_music", "arguments": {"song": "Take Five"}}],
    },
    {
        "name": "weather_reykjavik",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "How's the weather in Reykjavik?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "Reykjavik"}}],
    },
    {
        "name": "alarm_noon",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Set an alarm for 12:00 PM."}],
        "tools": [TOOL_SET_ALARM],
        "expected_calls": [{"name": "set_alarm", "arguments": {"hour": 12, "minute": 0}}],
    },
    {
        "name": "reminder_water_plants",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Remind me to water the plants at 6:15 PM."}],
        "tools": [TOOL_CREATE_REMINDER],
        "expected_calls": [{"name": "create_reminder", "arguments": {"title": "water the plants", "time": "6:15 PM"}}],
    },

    # ===== Medium: 2-3 tools, must pick the right one =====
    {
        "name": "choose_message_over_alarm",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Send Chen a quick note: running 10 min late."}],
        "tools": [TOOL_SET_ALARM, TOOL_SEND_MESSAGE],
        "expected_calls": [{"name": "send_message", "arguments": {"recipient": "Chen", "message": "running 10 min late"}}],
    },
    {
        "name": "choose_weather_over_music",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Do I need a jacket in Madrid today?"}],
        "tools": [TOOL_PLAY_MUSIC, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "Madrid"}}],
    },
    {
        "name": "choose_timer_over_alarm",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Start a 25 minute countdown."}],
        "tools": [TOOL_SET_TIMER, TOOL_SET_ALARM, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "set_timer", "arguments": {"minutes": 25}}],
    },
    {
        "name": "choose_reminder_over_message",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Remind me to submit the report at 11:00 AM."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_CREATE_REMINDER, TOOL_PLAY_MUSIC],
        "expected_calls": [{"name": "create_reminder", "arguments": {"title": "submit the report", "time": "11:00 AM"}}],
    },
    {
        "name": "choose_search_contacts",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Look up Sofia in my contacts."}],
        "tools": [TOOL_SEARCH_CONTACTS, TOOL_GET_WEATHER, TOOL_SET_TIMER],
        "expected_calls": [{"name": "search_contacts", "arguments": {"query": "Sofia"}}],
    },
    {
        "name": "choose_play_music",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Put on some synthwave."}],
        "tools": [TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_ALARM],
        "expected_calls": [{"name": "play_music", "arguments": {"song": "synthwave"}}],
    },
    {
        "name": "choose_alarm_specific",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Wake me at 4:05 AM."}],
        "tools": [TOOL_SET_TIMER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE],
        "expected_calls": [{"name": "set_alarm", "arguments": {"hour": 4, "minute": 5}}],
    },
    {
        "name": "choose_weather_among_three",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "What's the weather like in Nairobi?"}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_CREATE_REMINDER],
        "expected_calls": [{"name": "get_weather", "arguments": {"location": "Nairobi"}}],
    },
    {
        "name": "choose_message_short",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Ping Mateo: call me when you're free."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_SEARCH_CONTACTS, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER],
        "expected_calls": [{"name": "send_message", "arguments": {"recipient": "Mateo", "message": "call me when you're free"}}],
    },
    {
        "name": "choose_reminder_evening",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Set a reminder to pick up laundry at 7:20 PM."}],
        "tools": [TOOL_SET_ALARM, TOOL_CREATE_REMINDER, TOOL_SET_TIMER, TOOL_PLAY_MUSIC],
        "expected_calls": [{"name": "create_reminder", "arguments": {"title": "pick up laundry", "time": "7:20 PM"}}],
    },

    # ===== Hard: multiple tools needed, multi-call =====
    {
        "name": "weather_and_timer",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Check the weather in Singapore and set a 8 minute timer."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SET_TIMER, TOOL_SEND_MESSAGE],
        "expected_calls": [
            {"name": "get_weather", "arguments": {"location": "Singapore"}},
            {"name": "set_timer", "arguments": {"minutes": 8}},
        ],
    },
    {
        "name": "alarm_and_message",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Set an alarm for 5:50 AM and text Nora: don't forget your keys."}],
        "tools": [TOOL_SET_ALARM, TOOL_SEND_MESSAGE, TOOL_PLAY_MUSIC],
        "expected_calls": [
            {"name": "set_alarm", "arguments": {"hour": 5, "minute": 50}},
            {"name": "send_message", "arguments": {"recipient": "Nora", "message": "don't forget your keys"}},
        ],
    },
    {
        "name": "search_then_message",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Find Lina in my contacts and send: are you near the office?"}],
        "tools": [TOOL_SEARCH_CONTACTS, TOOL_SEND_MESSAGE, TOOL_GET_WEATHER],
        "expected_calls": [
            {"name": "search_contacts", "arguments": {"query": "Lina"}},
            {"name": "send_message", "arguments": {"recipient": "Lina", "message": "are you near the office?"}},
        ],
    },
    {
        "name": "music_and_reminder",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Play chillhop and remind me to start cooking at 6:00 PM."}],
        "tools": [TOOL_PLAY_MUSIC, TOOL_CREATE_REMINDER, TOOL_SET_TIMER, TOOL_SET_ALARM],
        "expected_calls": [
            {"name": "play_music", "arguments": {"song": "chillhop"}},
            {"name": "create_reminder", "arguments": {"title": "start cooking", "time": "6:00 PM"}},
        ],
    },
    {
        "name": "message_weather_alarm_three",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Text Diego 'see you at 8', check the weather in Lisbon, and set an alarm for 7:10 AM."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEARCH_CONTACTS],
        "expected_calls": [
            {"name": "send_message", "arguments": {"recipient": "Diego", "message": "see you at 8"}},
            {"name": "get_weather", "arguments": {"location": "Lisbon"}},
            {"name": "set_alarm", "arguments": {"hour": 7, "minute": 10}},
        ],
    },
    {
        "name": "timer_music_alarm",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Start a 30 minute timer, play focus music, and set an alarm for 3:05 PM."}],
        "tools": [TOOL_SET_TIMER, TOOL_PLAY_MUSIC, TOOL_SET_ALARM, TOOL_GET_WEATHER],
        "expected_calls": [
            {"name": "set_timer", "arguments": {"minutes": 30}},
            {"name": "play_music", "arguments": {"song": "focus music"}},
            {"name": "set_alarm", "arguments": {"hour": 15, "minute": 5}},
        ],
    },
    {
        "name": "reminder_search_message",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Look up Anya in my contacts, remind me to send the invoice at 4:40 PM, and message Anya: invoice going out soon."}],
        "tools": [TOOL_SEARCH_CONTACTS, TOOL_CREATE_REMINDER, TOOL_SEND_MESSAGE, TOOL_PLAY_MUSIC],
        "expected_calls": [
            {"name": "search_contacts", "arguments": {"query": "Anya"}},
            {"name": "create_reminder", "arguments": {"title": "send the invoice", "time": "4:40 PM"}},
            {"name": "send_message", "arguments": {"recipient": "Anya", "message": "invoice going out soon"}},
        ],
    },
    {
        "name": "weather_music_two",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "What's the weather in Kyoto, and play traditional instrumentals."}],
        "tools": [TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER],
        "expected_calls": [
            {"name": "get_weather", "arguments": {"location": "Kyoto"}},
            {"name": "play_music", "arguments": {"song": "traditional instrumentals"}},
        ],
    },
    {
        "name": "alarm_reminder_weather",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Set an alarm for 6:00 AM, remind me to pack snacks at 6:30 AM, and check the weather in Montreal."}],
        "tools": [TOOL_SET_ALARM, TOOL_CREATE_REMINDER, TOOL_GET_WEATHER, TOOL_SEND_MESSAGE],
        "expected_calls": [
            {"name": "set_alarm", "arguments": {"hour": 6, "minute": 0}},
            {"name": "create_reminder", "arguments": {"title": "pack snacks", "time": "6:30 AM"}},
            {"name": "get_weather", "arguments": {"location": "Montreal"}},
        ],
    },
    {
        "name": "search_message_timer",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Find Ravi in my contacts, message him 'joining in 5', and set a 5 minute timer."}],
        "tools": [TOOL_SEARCH_CONTACTS, TOOL_SEND_MESSAGE, TOOL_SET_TIMER, TOOL_PLAY_MUSIC],
        "expected_calls": [
            {"name": "search_contacts", "arguments": {"query": "Ravi"}},
            {"name": "send_message", "arguments": {"recipient": "Ravi", "message": "joining in 5"}},
            {"name": "set_timer", "arguments": {"minutes": 5}},
        ],
    },
]


def _normalize(v):
    """Normalize a value for comparison."""
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _call_matches(predicted, expected):
    """Check if a predicted call matches an expected call (name + argument values)."""
    if predicted["name"] != expected["name"]:
        return False
    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})
    for key, exp_val in exp_args.items():
        if key not in pred_args:
            return False
        if _normalize(pred_args[key]) != _normalize(exp_val):
            return False
    return True


def compute_f1(predicted_calls, expected_calls):
    """Compute F1 score between predicted and expected function calls."""
    if not predicted_calls and not expected_calls:
        return 1.0
    if not predicted_calls or not expected_calls:
        return 0.0

    matched = 0
    used = set()
    for exp in expected_calls:
        for i, pred in enumerate(predicted_calls):
            if i not in used and _call_matches(pred, exp):
                matched += 1
                used.add(i)
                break

    precision = matched / len(predicted_calls)
    recall = matched / len(expected_calls)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_benchmark(benchmarks=None):
    """Run all benchmark cases and print results."""
    if benchmarks is None:
        benchmarks = BENCHMARKS

    total = len(benchmarks)
    results = []
    for i, case in enumerate(benchmarks, 1):
        print(f"[{i}/{total}] Running: {case['name']} ({case['difficulty']})...", end=" ", flush=True)
        result = generate_hybrid(case["messages"], case["tools"])
        f1 = compute_f1(result["function_calls"], case["expected_calls"])
        source = result.get("source", "unknown")
        print(f"F1={f1:.2f} | {result['total_time_ms']:.0f}ms | {source}")
        results.append({
            "name": case["name"],
            "difficulty": case["difficulty"],
            "total_time_ms": result["total_time_ms"],
            "f1": f1,
            "source": source,
            "predicted": result["function_calls"],
            "expected": case["expected_calls"],
        })

    print("\n=== Benchmark Results ===\n")
    print(f"  {'#':>2} | {'Difficulty':<10} | {'Name':<28} | {'Time (ms)':>10} | {'F1':>5} | Source")
    print(f"  {'--':>2}-+-{'-'*10}-+-{'-'*28}-+-{'-'*10}-+-{'-'*5}-+-{'-'*20}")
    for i, r in enumerate(results, 1):
        print(f"  {i:>2} | {r['difficulty']:<10} | {r['name']:<28} | {r['total_time_ms']:>10.2f} | {r['f1']:>5.2f} | {r['source']}")

    print(f"\n--- Summary ---")
    for difficulty in ["easy", "medium", "hard"]:
        group = [r for r in results if r["difficulty"] == difficulty]
        if not group:
            continue
        avg_f1 = sum(r["f1"] for r in group) / len(group)
        avg_time = sum(r["total_time_ms"] for r in group) / len(group)
        on_device = sum(1 for r in group if r["source"] == "on-device")
        cloud = len(group) - on_device
        print(
            f"  {difficulty:<8} avg F1={avg_f1:.2f}  avg time={avg_time:.2f}ms  on-device={on_device}/{len(group)} cloud={cloud}/{len(group)}"
        )

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_time = sum(r["total_time_ms"] for r in results) / len(results)
    total_time = sum(r["total_time_ms"] for r in results)
    on_device_total = sum(1 for r in results if r["source"] == "on-device")
    cloud_total = len(results) - on_device_total
    print(f"  {'overall':<8} avg F1={avg_f1:.2f}  avg time={avg_time:.2f}ms  total time={total_time:.2f}ms")
    print(
        f"           on-device={on_device_total}/{len(results)} ({100*on_device_total/len(results):.0f}%)  cloud={cloud_total}/{len(results)} ({100*cloud_total/len(results):.0f}%)"
    )

    # Total score
    score = compute_total_score(results)
    print(f"\n{'='*50}")
    print(f"  TOTAL SCORE: {score:.1f}%")
    print(f"{'='*50}")

    return results


def compute_total_score(results):
    """
    Compute a total score from 0-100% as a weighted sum across difficulty levels.

    Components (per difficulty level):
      - F1 score (50%): accuracy of tool calls
      - Time score (25%): faster is better, capped at 500ms baseline
      - On-device ratio (25%): higher on-device usage is better

    Difficulty weights:
      - easy: 20%
      - medium: 30%
      - hard: 50%
    """
    difficulty_weights = {"easy": 0.20, "medium": 0.30, "hard": 0.50}
    time_baseline_ms = 500  # anything under this gets full marks

    total_score = 0
    for difficulty, weight in difficulty_weights.items():
        group = [r for r in results if r["difficulty"] == difficulty]
        if not group:
            continue

        avg_f1 = sum(r["f1"] for r in group) / len(group)
        avg_time = sum(r["total_time_ms"] for r in group) / len(group)
        on_device_ratio = sum(1 for r in group if r["source"] == "on-device") / len(group)

        time_score = max(0, 1 - avg_time / time_baseline_ms)

        level_score = (0.60 * avg_f1) + (0.15 * time_score) + (0.25 * on_device_ratio)
        total_score += weight * level_score

    return total_score * 100


if __name__ == "__main__":
    run_benchmark()
