# Final public corpus — schema and provenance

## Corpus inventory

One canonical release: `data/ling-dataset-all-600.jsonl` (600 JSONL rows)

| Split | Count | Per action | Per input mode |
|---|---:|---:|---:|
| Train | 480 | 96 | 120 |
| Validation | 60 | 12 | 15 |
| Test | 60 | 12 | 15 |
| **Total** | **600** | **120** | **150** |

Action set: `list_files`, `read_file`, `search_files`, `text_retrieval`, `reject_none`

Modes: `zh`, `en`, `mixed`, `machine` (JSON-like / Python-like / XML-like prompt surface variants)

Each row contains `id`, `split`, `category`, `messages`, `target`, `metadata`. The `target` is `{"name":"...","arguments":{...}}`. The assistant turn in `messages` contains the canonical serialized JSON target. `metadata.sample_sha256` covers the three `messages` entries, not the `id`, so replacing obsolete release-number ID prefixes leaves the content-hash invariant unchanged.

The corpus is an original balanced synthetic dataset, not a published BFCL leaderboard result. It varies the input language and tool-definition formats and retains JSON as the primary response contract. The separately designed 180-sample return-format suite (JSON/Python/XML) is **not included in this minimal PR**.

### Public corpus tool parameters

- `list_files(path: str, recursive?: bool)`
- `read_file(path: str, offset?: int, limit?: int)`
- `search_files(pattern: str, scope?: str)`
- `text_retrieval(query: str, top_k?: int)`
- `reject_none(reason: str)`

Tool names, parameter keys, reason codes and code/path literals are not translated. Natural-language query values preserve user language unless a task explicitly requests translation. Syntax punctuation for JSON is required; natural-language punctuation is varied intentionally.

### Static validation

```bash
python validate-ling-dataset.py
```

The validator checks 600 rows, split/action/mode counts, unique IDs and within-split prompts, target-assistant serialization, and per-message SHA-256. A `seed-targets.json` metadata file retains the 150 original seed objectives but is **not a second training corpus**.

### Experiment-line distinction

The physical LoRA weights and the 60-case Base/LoRA traces were produced by a **different deterministic data generator embedded in `run-ling-training.py`**, with `{"tool":"...","arguments":{...}}` output and training-time tool parameters such as `directory`. This published 600-row JSONL uses `{"name":"...","arguments":{...}}`, for independent design, auditing and later research reuse. These contracts must not be conflated. No claim is made that the published corpus reproduces the recorded 60-case scores.
