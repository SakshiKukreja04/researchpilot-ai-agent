# Evaluation

This file is a record template for actual runs. It must not be used to claim
live success without inspecting the generated result file.

## Cases

The ten evaluation cases are defined in
[evaluation/test_cases.json](evaluation/test_cases.json). They cover three
normal research questions, finite-list and security questions, malformed input,
comparisons, prompt-injection handling, and an intentionally obscure question
that should not lead to fabricated evidence.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe evaluation\run_eval.py --mode deterministic
.\.venv\Scripts\python.exe evaluation\run_eval.py --mode live
```

The live runner records the model used by the execution trace, including the
fallback model when synthesis fails; it never records the API key. Results are
saved locally under `evaluation/results/` and are intentionally ignored by Git
because live outputs may contain external page content.

## Interpretation

The runner reports completion rate and citation validity against the retrieved
source registry. Citation validity means that a citation ID exists; it does not
prove that a claim is factually correct. Human review of cited excerpts is
required for claim-support conclusions. Latency and failure causes should be
reported from the recorded trace rather than invented or rounded.

## Prototype verification run

The project virtual environment (`.venv`, Python 3.12.10) was
used to run:

```text
python -m pip check        -> No broken requirements found
python -m pytest -q        -> 155 passed
```

The deterministic evaluator runs all ten cases. Fixture-backed research cases
return controlled answers with valid citation references; whitespace input is
rejected before external calls; and obscure-topic cases do not fabricate
evidence. These are deterministic pipeline checks, not live research-quality
measurements.

The default live configuration is:

```dotenv
GROQ_MODEL=openai/gpt-oss-120b
GROQ_SYNTHESIS_FALLBACK_MODEL=openai/gpt-oss-20b
GROQ_MAX_OUTPUT_TOKENS=4096
GROQ_REASONING_EFFORT=low
```

Retrieval uses bounded thread-based waves. At most `FETCH_MAX_WORKERS` page
fetches run concurrently, while candidate ordering, the page budget, SSRF
checks, evidence scoring, and output ordering remain deterministic. Retrieval
can stop after repeated failures or when the first wave has no relevant page;
set either early-stop setting to `0` to disable that rule.
Search uses the same bounded worker setting. A whole search operation is capped
by `SEARCH_QUERY_TIMEOUT_SECONDS`, including primary, fallback, and simplified
retry. `SEARCH_BUDGET_FRACTION` and `MIN_FETCH_WINDOW_SECONDS` reserve time for
fetching and synthesis. If search reaches its budget with candidate URLs
available, the run records `search_deadline_reached` and continues to
retrieval. Deadline-caused no-source outcomes use status `partial` with reason
code `deadline_exhausted_during_search` or `deadline_exhausted_during_fetch`;
they are not mislabeled as ordinary insufficient web evidence.
Trace timestamps represent event completion times. Aggregate latency is measured
from the pipeline start/end timer rather than summing overlapping event durations.

The main tradeoffs are lower latency without increasing the page budget or
provider cost, at the expense of slightly burstier traffic and more detailed
wave tracing. Early stopping saves time on unanswerable queries but can miss a
late relevant source; conservative thresholds and first-wave-only relevance
checks mitigate that risk.

## Offline latency benchmark

The benchmark uses six deterministic 0.2-second fake page fetches. It writes
the complete machine-readable result to
[`evaluation/results/latency_benchmark.json`](evaluation/results/latency_benchmark.json).

| Workers | Wall time (s) | Speedup vs 1 | Fetch attempts | Accepted sources |
|---:|---:|---:|---:|---:|
| 1 | 1.2477 | 1.00x | 6 | 6 |
| 2 | 0.6341 | 1.97x | 6 | 6 |
| 3 | 0.4250 | 2.94x | 6 | 6 |
| 4 | 0.4275 | 2.92x | 6 | 6 |

These are offline measurements on the local development environment, not
claims about live provider latency. Use `latency_benchmark.py --live` only
with a configured Groq key to compare E1 and E10 against live providers.

The primary and fallback model IDs should be checked against the current Groq
catalog before a live run. If a model is unavailable, the application reports
an actionable configuration error pointing to
<https://console.groq.com/docs/models> and attempts the configured synthesis
fallback.
