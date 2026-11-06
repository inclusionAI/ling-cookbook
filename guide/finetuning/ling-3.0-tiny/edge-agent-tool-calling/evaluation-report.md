# Ling-3.0-tiny — Edge Agent Tool Calling: recorded Base vs QLoRA evaluation

**Experimental basis:** Google Colab NVIDIA L4; `inclusionAI/Ling-3.0-tiny`; 4-bit NF4 LoRA r=8, alpha=16, 3 epochs, 90 logged steps. Historical training elapsed = 3406.5 seconds. Final recorded loss at step 90 = 0.1693; first logged value at step 10 = 2.3435 (92.8% reduction from step 10 to step 90).

## Recomputed results from raw outputs

| Metric | Base | QLoRA | Change |
|---|---:|---:|---:|
| First complete tool-call JSON parseability | 55/60 = 91.7% | **60/60 = 100.0%** | +8.3 percentage points |
| Tool routing (correct action identifier) | 53/60 = 88.3% | **59/60 = 98.3%** | +10.0 percentage points |
| Rejection-case hit rate (8 designated cases) | 5/8 = 62.5% | **8/8 = 100.0%** | +37.5 percentage points |

These figures were recomputed independently from raw decoded strings in `evidence/recorded-traces.json` using `evaluate-traces.py` without trusting the original trace metadata's pre-fix score fields.

### What each metric means

- **First-object JSON parseability:** find the first complete JSON object in the raw response; count it only when it has `tool` (string) and `arguments` (object)
- **Routing:** the extracted `tool` equals `expected_tool` for the same recorded case
- **Rejection-case hit rate:** of 8 cases whose expected action is `reject_none`, count successful selection of that action. This is a conditional success rate / recall, **not precision**
- **Full-response JSON-only validity:** distinct from first-object parseability and not asserted to be 100%; generated material can follow the tool-call object
- **Parameter exact match and schema compliance:** not asserted from this trace-only reevaluation, because the archived traces retain expected action labels but not each case's complete reference argument dictionary. Do not equate tool routing with full-call exact match

The final output sometimes contains a first valid tool-call JSON followed by `<|im_end|>` and later generated dialogue. The original scorecard was affected by stop-token / full-response parsing assumptions. The revised score protocol evaluates the first complete tool-call object and preserves the raw original output for audit.

The comparison applies to the **recorded training-script's 60 test examples**. The independent published 600-row corpus is NOT the source of the 60 original run outputs. No claim is made that the full 600-row corpus was used to train the physical LoRA weights. The model never executed tools; therefore none of these are end-to-end agent safety or runtime reliability metrics.

## Artifact boundaries

- **Physical LoRA archive:** `adapter.tar.gz`, hosted in the contributor's fork repository (`lostlight530/ling-cookbook` main branch — a standalone GitHub repository), separate from the upstream PR diff
- **Raw generation records:** `evidence/recorded-traces.json`, 60 QLoRA + 60 Base model outputs
- **CPU verification:** `python evaluate-traces.py`
- **Training source:** `run-ling-training.py` with unchanged seed-42 sample generation/training hyperparameters and a corrected post-training evaluation section
