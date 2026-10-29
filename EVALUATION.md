# Evaluation

This file is a record template for actual runs. It must not be used to claim
live success without inspecting the generated result file.

## Cases

The five prototype cases are defined in
[evaluation/test_cases.json](evaluation/test_cases.json). They cover three
normal research questions, empty input, and an intentionally obscure question
that should not lead to fabricated evidence.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe evaluation\run_eval.py --mode deterministic
.\.venv\Scripts\python.exe evaluation\run_eval.py --mode live
```

The live runner records the configured model indirectly through the application
trace only if it is exposed by the current implementation; it never records
the API key. Results are saved locally under `evaluation/results/` and are
intentionally ignored by Git because live outputs may contain external page
content.

## Interpretation

The runner reports completion rate and citation validity against the retrieved
source registry. Citation validity means that a citation ID exists; it does not
prove that a claim is factually correct. Human review of cited excerpts is
required for claim-support conclusions. Latency and failure causes should be
reported from the recorded trace rather than invented or rounded.

## Prototype verification run

On 2026-10-29, the project virtual environment (`.venv`, Python 3.12.10) was
used to run:

```text
python -m pip check        -> No broken requirements found
python -m pytest -q        -> 140 passed
```

The deterministic evaluator ran all five cases. The three fixture-backed
research cases returned controlled `partial` answers with four valid citation
references; the whitespace case was rejected before external calls; and the
obscure-topic case did not fabricate evidence. These are deterministic pipeline
checks, not live research-quality measurements.

The real `web_search` tool was also exercised with the four research topics.
It returned results for the PostgreSQL/SQLite, SSRF, and obscure-topic queries;
the RAG query returned zero results on that attempt. Full live end-to-end runs
were not possible because the local `.env` still contains the placeholder Groq
API key. No live Groq result is claimed.
