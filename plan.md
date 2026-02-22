# Plan: Merge sneha improvements into main & optimize score

## Current State

### Sneha branch (current): 82.5% total score
- F1: 100% on all 30 tests
- Time: avg 1097ms (well above 500ms baseline → time score = 0)
- On-device: 28/30 (93%) — 2 hard cases fall back to cloud

### Main branch (previous): 85.8% total score
- Same core regex-hybrid logic, no query rewriting or traffic shifter

### Score formula
- F1 (60% weight) + Time score (15% weight) + On-device ratio (25% weight)
- Weighted by difficulty: easy 20%, medium 30%, hard 50%

## What's good in sneha (KEEP)
1. **Query rewriting** (`REWRITE_RULES` + `_rewrite_query`) — normalizes natural language to canonical forms FunctionGemma handles better (e.g., "wake me up at" → "set an alarm for")

## What's bad in sneha (DROP)
1. **TrafficShifter** — introduces `random.random()` non-determinism, can route queries to cloud unnecessarily, hurts on-device ratio without benefit since local+regex already achieves 100% F1
2. **Intent counting via keywords** — already handled by `_is_multi_request()`

## Steps

### Step 1: Switch to main, create feature branch
- `git checkout main && git checkout -b feature/merge-sneha-improvements`

### Step 2: Cherry-pick query rewriting from sneha
- Add `REWRITE_RULES`, `_rewrite_query()` to main.py
- Use rewritten messages for cactus calls (not for regex extraction — regex should use original user text)

### Step 3: Fix the 2 cloud-fallback hard cases on-device
- `alarm_and_reminder`: "Set an alarm for 6:45 AM and remind me to take medicine at 7:00 AM"
  - Issue: `_extract_time()` finds first time only, both tools get 6:45 AM
  - Fix: Extract multiple times from text, assign them to tools based on position/context
- `timer_music_reminder`: "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM"
  - Three tools needed — may need smarter multi-tool regex decomposition
  - Fix: Split compound requests by "and"/"," and match each segment to its tool

### Step 4: Optimize cactus model loading for speed
- Cache the model globally instead of `cactus_init` + `cactus_destroy` per call
- This can save ~100-200ms per call (model load overhead)

### Step 5: Fix run_benchmark.py paths
- Already done (fixed cactus symlink + hackathon dir path)

### Step 6: Run benchmark, verify improvement over 85.8%

### Step 7: Commit and create PR
