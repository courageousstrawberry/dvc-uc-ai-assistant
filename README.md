# DVC → UC Transfer Assistant — Demo (transfer-assistant demo)

This repository contains a demo chatbot that helps students map Diablo Valley College (DVC) courses to University of California (UC) courses and transfer requirements. It is a simplified demo of the full project: https://github.com/dvc-uc-ai-assistant/transfer-assistant.

This workspace is intentionally lightweight and focused on demonstrating:
- Grounded answers using a compact, local JSON dataset of articulation agreements (agreements_25-26/)
- Structured intent extraction (JSON) from natural language queries
- Deterministic listing of transfer requirements for one or more UC campuses

This README explains how the demo works, how to run it locally, and where to look in the code.

## What's in this repo

- `src/ai_agent_vLLM.py` — The demo chat agent. Key features:
	- Extracts intent as a compact JSON (`campus_keys`, `categories`, `uc_course_codes`, `dvc_course_codes`, `action`) using an LLM.
	- Builds a deterministic, grounded `CONTEXT` string from local JSON course maps.
	- For "requirements" queries, prints exhaustive DVC→UC mappings for requested campus(es).
	- For other queries, sends the `CONTEXT` plus question to the chat model to generate a grounded answer.

- `src/extract_course_list.py` — Loads the JSON articulation files in `agreements_25-26/` and builds in-memory maps (`ucsd_map`, `ucb_map`, etc.) the agent uses to format context.

- `agreements_25-26/` — JSON articulation agreements (example: `ucsd_25-26_cs.json`) used as the ground truth for responses.

- `.env.example` — Example environment variables; copy to `.env` and add your OpenAI API key.

- `requirements.txt` — Python dependencies (python-dotenv, openai, etc.).

## Quick start (Windows / PowerShell)

1. Create or activate your conda environment (example):

```powershell
conda activate sklearn-env
```

2. Install dependencies (if not already installed):

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file at the repository root with your OpenAI API key:

```text
OPENAI_API_KEY=sk-...
```

Or copy the example:

```powershell
copy .env.example .env
# then edit .env to add your key
```

4. Run the interactive demo:

```powershell
python src/ai_agent_vLLM.py
```

5. Non-interactive example (Windows `cmd` style piping used in tests):

```powershell
cmd /c "(echo what are the ucsd and ucb transfer requirements? & echo exit) | C:\Path\To\python.exe src\ai_agent_vLLM.py"
```

Replace `C:\Path\To\python.exe` with your Python executable path if needed.

## Example queries

- "What are the UCSD transfer requirements for Computer Science?"
- "What DVC courses map to UC Berkeley COMPSCI-61B?"
- "Do you accept COMSC-210 for UCLA?" (note: demo currently filters out UCLA/UCI from printed requirement lists by default — see `build_context_from_intent`)

## Design highlights

- Strict grounding: the assistant only uses the content present in the JSON files to build the `CONTEXT`. When asked about requirements, it prints deterministic mappings and avoids hallucination.

- Intent format: the LLM returns a compact JSON with fields that align to the local JSON naming (e.g., `San_Diego` → `UCSD`) so the mapping code is straightforward.

- Multi-campus support: `build_context_from_intent` can detect multiple campuses in the user query (or use the extracted `campus_keys`) and concatenates context for each campus.

- Deterministic requirements path: to guarantee exhaustive listings for "requirements" queries, the agent bypasses the model and directly returns the formatted `CONTEXT`.

## File pointers (where to change behavior)

- `src/ai_agent_vLLM.py`
	- `INTENT_SYSTEM_PROMPT` — modifies the JSON schema and alias rules given to the model.
	- `extract_intent_simple()` — the function that calls the model to parse intent.
	- `build_context_from_intent()` — maps campuses and calls `format_from_map()`; tweak line budgets or campus filters here.
	- `format_from_map()` — controls the textual layout for each DVC→UC mapping.
	- `ANSWER_SYSTEM_PROMPT` — controls how the assistant is instructed to answer when the model is used.

- `src/extract_course_list.py` — if you have new JSON agreements, add them under `agreements_25-26/` and create a mapping line in this module.

## Limitations and TODOs

- This is a demo: it intentionally contains simplified parsing and output rules to make the behavior predictable.
- The demo currently filters out UCLA and UCI from printed requirement lists by default — adjust `build_context_from_intent()` to change this.
- There are no unit tests provided; adding tests for `format_from_map()` and `map_uc_dvc()` would be useful.

## Contributing / Extending

If you want to extend the demo into a production-ready assistant (like the full transfer-assistant), consider:
- Adding unit tests and CI.
- Implementing function-calling or stricter schema validation for intent extraction.
- Adding a web UI and authentication.
- Preserving provenance (which JSON file and which version of an articulation was used for each mapping).

## License

This demo follows the repository license in `LICENSE`.

---

This README was added as a demo description for the repository and highlights how the `ai_agent_vLLM.py` script demonstrates grounding and multi-campus requirements listing. If you want it to mention specific changes made in your fork or include usage examples capturing the exact outputs, tell me and I can add sample outputs or adjust wording to your preference.
