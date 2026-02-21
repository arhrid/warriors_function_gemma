import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json
from main import generate_hybrid


############## Tool definitions (NEW) ##############

TOOL_GET_FORECAST = {
    "name": "get_forecast",
    "description": "Get the weather forecast for a specific location and date.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and state/country, e.g., 'Austin, TX'"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
        },
        "required": ["location", "date"],
    },
}

TOOL_CONVERT_CURRENCY = {
    "name": "convert_currency",
    "description": "Convert an amount of money from one currency to another.",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "3-letter currency code, e.g., 'USD'"},
            "to_currency": {"type": "string", "description": "3-letter currency code, e.g., 'JPY'"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
}

TOOL_BOOK_RIDE = {
    "name": "book_ride",
    "description": "Book a ride with pickup and dropoff locations at a given time.",
    "parameters": {
        "type": "object",
        "properties": {
            "pickup": {"type": "string", "description": "Pickup address or place"},
            "dropoff": {"type": "string", "description": "Dropoff address or place"},
            "time": {"type": "string", "description": "Pickup time, e.g., 'now' or '2026-02-22 08:15'"},
            "ride_type": {"type": "string", "description": "One of: economy, comfort, xl"},
        },
        "required": ["pickup", "dropoff", "time", "ride_type"],
    },
}

TOOL_CREATE_CAL_EVENT = {
    "name": "create_calendar_event",
    "description": "Create a calendar event with time window, location, and attendees.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title"},
            "start_time": {"type": "string", "description": "Start timestamp, e.g., '2026-03-01 14:00'"},
            "end_time": {"type": "string", "description": "End timestamp, e.g., '2026-03-01 15:00'"},
            "location": {"type": "string", "description": "Event location"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee emails"},
        },
        "required": ["title", "start_time", "end_time"],
    },
}

TOOL_ADD_TO_SHOPPING_LIST = {
    "name": "add_to_shopping_list",
    "description": "Add one or more items to a named shopping list.",
    "parameters": {
        "type": "object",
        "properties": {
            "list_name": {"type": "string", "description": "Shopping list name, e.g., 'Groceries'"},
            "items": {"type": "array", "items": {"type": "string"}, "description": "Items to add"},
        },
        "required": ["list_name", "items"],
    },
}

TOOL_SET_FOCUS_MODE = {
    "name": "set_focus_mode",
    "description": "Enable a focus mode for a duration and optionally allow certain contacts to bypass.",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "description": "One of: deep_work, driving, sleep"},
            "duration_minutes": {"type": "integer", "description": "How long to enable the mode"},
            "allow_calls_from": {"type": "array", "items": {"type": "string"}, "description": "Names who can bypass"},
        },
        "required": ["mode", "duration_minutes"],
    },
}

TOOL_TRANSLATE_TEXT = {
    "name": "translate_text",
    "description": "Translate text into a target language.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "target_language": {"type": "string", "description": "Target language, e.g., 'Spanish'"},
            "style": {"type": "string", "description": "Optional style hint, e.g., 'formal' or 'casual'"},
        },
        "required": ["text", "target_language"],
    },
}


############## Benchmark cases (NEW) ##############

BENCHMARKS = [
    # ===== Easy: 1 tool, direct request =====
    {
        "name": "forecast_austin_tomorrow",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "What's the forecast for Austin, TX tomorrow (2026-02-22)?"}],
        "tools": [TOOL_GET_FORECAST],
        "expected_calls": [{"name": "get_forecast", "arguments": {"location": "Austin, TX", "date": "2026-02-22"}}],
    },
    {
        "name": "convert_50_usd_to_eur",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Convert 50 dollars to euros."}],
        "tools": [TOOL_CONVERT_CURRENCY],
        "expected_calls": [{"name": "convert_currency", "arguments": {"amount": 50, "from_currency": "USD", "to_currency": "EUR"}}],
    },
    {
        "name": "ride_now_home_to_airport",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Book me an economy ride now from 12 Market St to SFO."}],
        "tools": [TOOL_BOOK_RIDE],
        "expected_calls": [{"name": "book_ride", "arguments": {"pickup": "12 Market St", "dropoff": "SFO", "time": "now", "ride_type": "economy"}}],
    },
    {
        "name": "add_groceries_basic",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Add milk, bananas, and oats to my Groceries list."}],
        "tools": [TOOL_ADD_TO_SHOPPING_LIST],
        "expected_calls": [{"name": "add_to_shopping_list", "arguments": {"list_name": "Groceries", "items": ["milk", "bananas", "oats"]}}],
    },
    {
        "name": "focus_deep_work_45",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Turn on deep work focus for 45 minutes."}],
        "tools": [TOOL_SET_FOCUS_MODE],
        "expected_calls": [{"name": "set_focus_mode", "arguments": {"mode": "deep_work", "duration_minutes": 45}}],
    },
    {
        "name": "translate_to_spanish",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Translate 'Where is the nearest pharmacy?' to Spanish."}],
        "tools": [TOOL_TRANSLATE_TEXT],
        "expected_calls": [{"name": "translate_text", "arguments": {"text": "Where is the nearest pharmacy?", "target_language": "Spanish"}}],
    },
    {
        "name": "create_event_simple",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Create a calendar event called 'Budget Review' on 2026-03-01 from 2pm to 3pm."}],
        "tools": [TOOL_CREATE_CAL_EVENT],
        "expected_calls": [{"name": "create_calendar_event", "arguments": {"title": "Budget Review", "start_time": "2026-03-01 14:00", "end_time": "2026-03-01 15:00"}}],
    },
    {
        "name": "forecast_tokyo_specific_date",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Forecast for Tokyo on 2026-02-25?"}],
        "tools": [TOOL_GET_FORECAST],
        "expected_calls": [{"name": "get_forecast", "arguments": {"location": "Tokyo", "date": "2026-02-25"}}],
    },
    {
        "name": "convert_1200_jpy_to_usd",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "How much is 1200 JPY in USD?"}],
        "tools": [TOOL_CONVERT_CURRENCY],
        "expected_calls": [{"name": "convert_currency", "arguments": {"amount": 1200, "from_currency": "JPY", "to_currency": "USD"}}],
    },
    {
        "name": "ride_comfort_sj_to_paloalto",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Schedule a comfort ride at 2026-02-22 09:10 from San Jose Diridon to Palo Alto Caltrain."}],
        "tools": [TOOL_BOOK_RIDE],
        "expected_calls": [{"name": "book_ride", "arguments": {"pickup": "San Jose Diridon", "dropoff": "Palo Alto Caltrain", "time": "2026-02-22 09:10", "ride_type": "comfort"}}],
    },

    # ===== Medium: 1 tool, indirect phrasing / mild ambiguity =====
    {
        "name": "forecast_with_rephrase",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Do I need an umbrella in Seattle on 2026-02-24?"}],
        "tools": [TOOL_GET_FORECAST],
        "expected_calls": [{"name": "get_forecast", "arguments": {"location": "Seattle", "date": "2026-02-24"}}],
    },
    {
        "name": "convert_spoken_amount",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "If I have 19.99 USD, what is that in CAD?"}],
        "tools": [TOOL_CONVERT_CURRENCY],
        "expected_calls": [{"name": "convert_currency", "arguments": {"amount": 19.99, "from_currency": "USD", "to_currency": "CAD"}}],
    },
    {
        "name": "shopping_list_synonyms",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Put toothpaste and floss on the shopping list called 'Errands'."}],
        "tools": [TOOL_ADD_TO_SHOPPING_LIST],
        "expected_calls": [{"name": "add_to_shopping_list", "arguments": {"list_name": "Errands", "items": ["toothpaste", "floss"]}}],
    },
    {
        "name": "focus_driving_with_bypass",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "I'm driving—enable driving focus for 90 minutes, but let calls from Mom through."}],
        "tools": [TOOL_SET_FOCUS_MODE],
        "expected_calls": [{"name": "set_focus_mode", "arguments": {"mode": "driving", "duration_minutes": 90, "allow_calls_from": ["Mom"]}}],
    },
    {
        "name": "translate_formal",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Translate 'Thank you for your patience' to French, formal tone."}],
        "tools": [TOOL_TRANSLATE_TEXT],
        "expected_calls": [{"name": "translate_text", "arguments": {"text": "Thank you for your patience", "target_language": "French", "style": "formal"}}],
    },
    {
        "name": "event_with_location_attendees",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Schedule 'Design Sync' on 2026-03-03 10:30-11:15 at Zoom and invite a@x.com and b@y.com."}],
        "tools": [TOOL_CREATE_CAL_EVENT],
        "expected_calls": [{"name": "create_calendar_event", "arguments": {
            "title": "Design Sync",
            "start_time": "2026-03-03 10:30",
            "end_time": "2026-03-03 11:15",
            "location": "Zoom",
            "attendees": ["a@x.com", "b@y.com"],
        }}],
    },
    {
        "name": "ride_xl_with_indirect",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "We’re 5 people—can you get us an XL from Union Square to Pier 39 at 2026-02-22 13:20?"}],
        "tools": [TOOL_BOOK_RIDE],
        "expected_calls": [{"name": "book_ride", "arguments": {"pickup": "Union Square", "dropoff": "Pier 39", "time": "2026-02-22 13:20", "ride_type": "xl"}}],
    },
    {
        "name": "forecast_with_country",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Weather outlook for Vancouver, Canada on 2026-02-23?"}],
        "tools": [TOOL_GET_FORECAST],
        "expected_calls": [{"name": "get_forecast", "arguments": {"location": "Vancouver, Canada", "date": "2026-02-23"}}],
    },
    {
        "name": "convert_rounding_noise",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Convert 3.5 GBP into USD please."}],
        "tools": [TOOL_CONVERT_CURRENCY],
        "expected_calls": [{"name": "convert_currency", "arguments": {"amount": 3.5, "from_currency": "GBP", "to_currency": "USD"}}],
    },
    {
        "name": "event_title_variation",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Add 'Parent-teacher conference' to my calendar: 2026-03-05 16:00 to 16:30, location: Lincoln Elementary."}],
        "tools": [TOOL_CREATE_CAL_EVENT],
        "expected_calls": [{"name": "create_calendar_event", "arguments": {
            "title": "Parent-teacher conference",
            "start_time": "2026-03-05 16:00",
            "end_time": "2026-03-05 16:30",
            "location": "Lincoln Elementary",
        }}],
    },

    # ===== Hard: 2-3 tools, planning / multi-step / context =====
    {
        "name": "trip_plan_weather_then_ride",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "For 2026-02-22, check the forecast for Napa, then book an economy ride from 200 Main St to Napa Valley Museum at 09:00."}
        ],
        "tools": [TOOL_GET_FORECAST, TOOL_BOOK_RIDE],
        "expected_calls": [
            {"name": "get_forecast", "arguments": {"location": "Napa", "date": "2026-02-22"}},
            {"name": "book_ride", "arguments": {"pickup": "200 Main St", "dropoff": "Napa Valley Museum", "time": "2026-02-22 09:00", "ride_type": "economy"}},
        ],
    },
    {
        "name": "budget_convert_then_event",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Convert 250 EUR to USD, and make a calendar event 'Pay contractor' on 2026-03-02 09:00-09:10."}
        ],
        "tools": [TOOL_CONVERT_CURRENCY, TOOL_CREATE_CAL_EVENT],
        "expected_calls": [
            {"name": "convert_currency", "arguments": {"amount": 250, "from_currency": "EUR", "to_currency": "USD"}},
            {"name": "create_calendar_event", "arguments": {"title": "Pay contractor", "start_time": "2026-03-02 09:00", "end_time": "2026-03-02 09:10"}},
        ],
    },
    {
        "name": "shopping_then_translate_note",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Add 'rice' and 'lentils' to Groceries, then translate 'I will be late by 10 minutes' to German."}
        ],
        "tools": [TOOL_ADD_TO_SHOPPING_LIST, TOOL_TRANSLATE_TEXT],
        "expected_calls": [
            {"name": "add_to_shopping_list", "arguments": {"list_name": "Groceries", "items": ["rice", "lentils"]}},
            {"name": "translate_text", "arguments": {"text": "I will be late by 10 minutes", "target_language": "German"}},
        ],
    },
    {
        "name": "focus_then_event",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Enable deep_work focus for 120 minutes and schedule 'Write report' from 2026-03-04 13:00 to 15:00."}
        ],
        "tools": [TOOL_SET_FOCUS_MODE, TOOL_CREATE_CAL_EVENT],
        "expected_calls": [
            {"name": "set_focus_mode", "arguments": {"mode": "deep_work", "duration_minutes": 120}},
            {"name": "create_calendar_event", "arguments": {"title": "Write report", "start_time": "2026-03-04 13:00", "end_time": "2026-03-04 15:00"}},
        ],
    },
    {
        "name": "forecast_two_cities",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Compare the forecast for Chicago on 2026-02-23 and Denver on 2026-02-24."}
        ],
        "tools": [TOOL_GET_FORECAST],
        "expected_calls": [
            {"name": "get_forecast", "arguments": {"location": "Chicago", "date": "2026-02-23"}},
            {"name": "get_forecast", "arguments": {"location": "Denver", "date": "2026-02-24"}},
        ],
    },
    {
        "name": "ride_then_focus_driving",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Book a comfort ride now from 1 Embarcadero Center to Oakland Airport, and switch on driving focus for 60 minutes."}
        ],
        "tools": [TOOL_BOOK_RIDE, TOOL_SET_FOCUS_MODE],
        "expected_calls": [
            {"name": "book_ride", "arguments": {"pickup": "1 Embarcadero Center", "dropoff": "Oakland Airport", "time": "now", "ride_type": "comfort"}},
            {"name": "set_focus_mode", "arguments": {"mode": "driving", "duration_minutes": 60}},
        ],
    },
    {
        "name": "event_with_translate_invite",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Create an event 'Team Lunch' on 2026-03-06 12:00-13:00 at 'SOMA Cafe' and then translate 'See you at noon!' to Italian."}
        ],
        "tools": [TOOL_CREATE_CAL_EVENT, TOOL_TRANSLATE_TEXT],
        "expected_calls": [
            {"name": "create_calendar_event", "arguments": {"title": "Team Lunch", "start_time": "2026-03-06 12:00", "end_time": "2026-03-06 13:00", "location": "SOMA Cafe"}},
            {"name": "translate_text", "arguments": {"text": "See you at noon!", "target_language": "Italian"}},
        ],
    },
    {
        "name": "currency_then_shopping_budget",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Convert 75 CAD to USD, then add 'gift wrap' to my Errands list."}
        ],
        "tools": [TOOL_CONVERT_CURRENCY, TOOL_ADD_TO_SHOPPING_LIST],
        "expected_calls": [
            {"name": "convert_currency", "arguments": {"amount": 75, "from_currency": "CAD", "to_currency": "USD"}},
            {"name": "add_to_shopping_list", "arguments": {"list_name": "Errands", "items": ["gift wrap"]}},
        ],
    },
    {
        "name": "travel_morning_sequence",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "On 2026-02-26, get the forecast for Boston; schedule 'Flight check-in' 2026-02-26 06:30-06:40; then book an economy ride at 2026-02-26 07:00 from 88 Summer St to BOS."}
        ],
        "tools": [TOOL_GET_FORECAST, TOOL_CREATE_CAL_EVENT, TOOL_BOOK_RIDE],
        "expected_calls": [
            {"name": "get_forecast", "arguments": {"location": "Boston", "date": "2026-02-26"}},
            {"name": "create_calendar_event", "arguments": {"title": "Flight check-in", "start_time": "2026-02-26 06:30", "end_time": "2026-02-26 06:40"}},
            {"name": "book_ride", "arguments": {"pickup": "88 Summer St", "dropoff": "BOS", "time": "2026-02-26 07:00", "ride_type": "economy"}},
        ],
    },
    {
        "name": "focus_sleep_then_translate",
        "difficulty": "hard",
        "messages": [
            {"role": "user", "content": "Set sleep focus for 480 minutes, and translate 'Good night, see you tomorrow' to Japanese (casual)."}
        ],
        "tools": [TOOL_SET_FOCUS_MODE, TOOL_TRANSLATE_TEXT],
        "expected_calls": [
            {"name": "set_focus_mode", "arguments": {"mode": "sleep", "duration_minutes": 480}},
            {"name": "translate_text", "arguments": {"text": "Good night, see you tomorrow", "target_language": "Japanese", "style": "casual"}},
        ],
    },
]


############## Scoring / runner (same structure) ##############

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


def compute_f1(pred_calls, exp_calls):
    """Compute F1 on tool calls (treat each expected call as a target)."""
    if not exp_calls and not pred_calls:
        return 1.0
    if not exp_calls and pred_calls:
        return 0.0
    if exp_calls and not pred_calls:
        return 0.0

    matched_exp = set()
    matched_pred = set()

    for i, p in enumerate(pred_calls):
        for j, e in enumerate(exp_calls):
            if j in matched_exp:
                continue
            if _call_matches(p, e):
                matched_exp.add(j)
                matched_pred.add(i)
                break

    tp = len(matched_pred)
    fp = len(pred_calls) - tp
    fn = len(exp_calls) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def run_benchmark():
    results = []
    print(f"Running {len(BENCHMARKS)} benchmark cases...\n")

    for idx, case in enumerate(BENCHMARKS, start=1):
        name = case["name"]
        difficulty = case["difficulty"]
        messages = case["messages"]
        tools = case["tools"]
        expected_calls = case["expected_calls"]

        result = generate_hybrid(messages, tools)

        pred_calls = result.get("function_calls", [])
        f1 = compute_f1(pred_calls, expected_calls)

        total_time_ms = result.get("total_time_ms", 0.0)
        source = "cloud (fallback)" if result.get("cloud_handoff") else "on-device"

        print(f"[{idx}/{len(BENCHMARKS)}] Running: {name} ({difficulty})... "
              f"F1={f1:.2f} | {total_time_ms:.0f}ms | {source}")

        results.append({
            "name": name,
            "difficulty": difficulty,
            "f1": f1,
            "total_time_ms": total_time_ms,
            "source": source,
        })

    total_score = compute_total_score(results)
    print("\n==============================")
    print(f"TOTAL SCORE: {total_score:.2f} / 100")
    print("==============================\n")
    return results


def compute_total_score(results):
    """Aggregate score across difficulty buckets."""
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
