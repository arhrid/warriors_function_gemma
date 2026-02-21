# Claude Code Instructions — Warriors Function Gemma

## Git Workflow

- Always `git fetch` before starting work.
- Stage with `git add .`, commit messages must be < 6 words.
- NEVER include `Co-Authored-By` or any coattribution in commits.
- Make atomic commits (one logical change per commit).
- Create a new branch and open a draft PR once you have a significant batch of related commits. Keep PR descriptions brief.

## Working Directory

All code changes happen inside `warriors_function_gemma/`. The `main.py` in the parent `functiongemma-hackathon/` folder is the original baseline — do NOT modify it.

## Task

Your main task is to modify the internal logic of the `generate_hybrid` method in `main.py`.

## Rules

- **Do not modify the input or output signature** (function arguments and return variables) of the `generate_hybrid` method. Keep the hybrid interface compatible with `benchmark.py`.
- Run `python benchmark.py` (from the parent `functiongemma-hackathon/` directory) to iterate, but your best score is preserved.
- The dataset is a hidden Cactus eval, quite difficult for FunctionGemma by design.

## Leaderboard Submission

Submit to the leaderboard with:

```
python submit.py --team "YourTeamName" --location "YourCity"
```

**Only submit 1x every 1 hour.** Do not spam submissions.
