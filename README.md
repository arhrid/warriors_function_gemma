# WarriorsPersonalAssistant
Personal voice agent: Handles WhatsApp calls/messages and sends email. Data is stored locally, using FunctionGemma to make tool calls.

The purpose of this document is to capture our usecase and the process. 

## Usecase Description
1. there is an active voice listening in our mac. 
2. The user would say "hey warrior, message xxx.xxx.xxxx number on my whatsapp that we are set for dinner at XYZ for 5pm today. Then, email his reponse to xxx@gmail.com.

## Our Architecture proposal for Internal logic of the generate_hybrid method in main.py
A multi-layered routing system that dynamically balances on-device (FunctionGemma) and cloud (Gemini) execution for optimal performance and accuracy.
Core Components
🚀 Model Persistence Optimization
pythonmodel = cactus_init(functiongemma_path)  # Load once at module level

🔄 Adaptive Traffic Shifter
Learns optimal routing per query category through exponential moving average:
pythoncategory = f"{intent_count}intent_{tool_type}"  # "2intent_weather"
success_rate = observe_outcomes()
local_prob += shift_rate * (success_rate - current_prob)
✍️ Query Rewriter
Normalizes indirect phrasing before processing:
python"wake me up at 8am" → "set an alarm for 8am"
"ping Alice" → "send a message to Alice saying"
🎯 Hybrid Argument Extraction

Tool Selection: FunctionGemma picks which tools to call
Argument Extraction: Regex deterministically extracts values

python# Model: {"name": "set_alarm", "arguments": {"hour": "ten"}}

```

## Execution Flow
```
Query → Rewrite → Traffic Decision → Local Processing → Regex Reconstruction → Cloud Fallback
                     ↓                    ↓                      ↓
              Skip if predicted      FunctionGemma +       Deterministic
              failure + complex      Tool Filtering        Arg Parsing
Performance Targets
Query TypeRoutingTimeOn-Device %SimpleLocal~150ms90%Multi-toolLocal Loop~300ms60%ComplexCloud Skip~400ms0%
Key Features

Domain-Agnostic: Works with any tool definitions via parameter-type extraction
Self-Adapting: Learns optimal routing from real performance outcomes
Performance Optimized: Model persistence + smart cloud skipping
High Accuracy: Regex reconstruction ensures correct argument parsing
