---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

##### Copyright 2026 Ant Group.

Thanks to [@lplalbert](https://github.com/lplalbert) for contributing this fine-tuning use case.

+++

# Fine-tuning Ling-3.0-tiny with MLX QLoRA to Play "Who Is the Undercover"

On Apple Silicon (verified on M4 MacBook Air 16GB), fine-tune
`inclusionAI/Ling-3.0-tiny` into a **player decision component** for the social
deduction game "Who Is the Undercover": given your role, your word and the round
history, produce a ≤25-character description and a vote (`我投N号` / `我弃权`).
The artifact doubles as the inference core of the **Apple MLX on-device model
engine** player bot for the [Avernet #2611](https://github.com/inclusionAI/Avernet/issues/2611)
game table — one project, two deliverables.

+++

## 1. What This Task Does

- **Data**: ~1k game records **programmatically synthesized** from 30 word pairs ×
  attribute pools × templates (reproducible, leakage-free, 6 held-out pairs for
  generalization), plus two increment tracks: live-game flywheel (undercover
  identity = ground-truth labels) and offline teacher distillation
  (**never enters the game loop**).
- **Training**: 4-bit base + LoRA (rank 8, last 16/24 layers, MoE experts frozen,
  104M/1.3% trainable), fully within 16GB unified memory.
- **Eval**: before/after comparison — utterance length compliance, word-leak rate,
  naturalness, vote format validity, vote accuracy, and **self-play win rate**
  (6 model seats + a scripted referee).
- **Application**: serve the model through `mlx_lm.server` (OpenAI-compatible),
  join the Avernet mixed table over the BCN V3 protocol, and play alongside
  other engines; output guards improve format reliability but do not certify game quality.

+++

## 2. Notes & Pitfalls

1. **Quantize locally**: the int4 pre-quant on ModelScope is vLLM-only; MLX needs
   `mlx_lm.convert` from the BF16 checkpoint.
2. **16G training boundary**: continuous QLoRA runs hit a deterministic
   `Metal Insufficient Memory` after ~625 steps — use **segmented training**
   (≤600 steps per segment, chained via `--resume-adapter-file`).
3. **Avoid quantized KV** (`--kv-bits 8`): the MLA path in mlx-lm used to crash on
   prompts >2048 tokens (tracked by the open upstream PR
   [#2058](https://github.com/ml-explore/mlx-lm/pull/2058)); a single 16G instance
   has plenty of KV headroom anyway.
4. **Weights are not in Git**: adapters are hosted on
   [GitHub Release](https://github.com/lplalbert/ling-cookbook/releases/tag/undercover-player-adapter-v2).
5. Greedy decoding collapses the model (canned speeches, all-abstain votes) — use
   seeded sampling (temp=0.7, top_p=0.9) everywhere.

+++

## 3. Model & Framework

- **Base**: `inclusionAI/Ling-3.0-tiny` (MoE, 1.3B active, `bailing_hybrid`,
  24 layers).
- **Framework**: Apple [MLX](https://github.com/ml-explore/mlx) + `mlx-lm`
  (QLoRA training and inference in one stack, `mask_prompt` loss on outputs only,
  gradient checkpointing).
- **LoRA topology**: attention projections on the last 16/24 layers only, MoE
  experts frozen — 104M / 1.3% trainable, fits 16GB for both training and a live
  game table.

+++

## 4. Requirements

| Item | Version / value |
|---|---|
| Hardware | Apple Silicon Mac (verified: M4 Air 16GB) |
| Python | ≥ 3.10 (verified: 3.13) |
| mlx-lm | ≥ 0.32.0 |
| Disk | BF16 15G + 4-bit 4.8G + adapter ~400M |
| Memory | ~12GB peak training; ~5.3GB serving single instance |

+++

## 5. Task Analysis

Each game has 6 players: 1 undercover and 5 civilians, each holding a similar
word. Every round: describe your word in ≤25 characters without leaking it, then
vote. The decision policy is split into two action heads:

1. **Speak**: role/word/history → one informative ≤25-char description;
2. **Vote**: same context → a ballot.

Prompts are **windowed**: history keeps round 1 + the latest 2 rounds (the middle
folds into `（第x-y轮略）`), so inference input is bounded (≤420 chars). Training,
inference and the flywheel share one choke point (`data/prompting.py`); the
Avernet player bot embeds a byte-identical renderer. Synthetic games span 1–6
rounds to match live depth, so the fold marker is covered in training
(626/1212 samples) — **any rule that folds/truncates input at inference time must
be covered across its full value range in training data**.

+++

## 6. Step-by-Step

<a id="setup"></a>

### Step 1: Environment & model quantization

```{code-cell}
%%bash
# Clone the repo and enter the recipe directory
git clone https://github.com/inclusionAI/ling-cookbook.git
cd ling-cookbook/guide/finetuning/ling-3.0-tiny/undercover-player

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download Ling-3.0-tiny (BF16, ~15G) to ~/models/Ling-3.0-tiny (ModelScope / HF),
# then quantize locally to 4-bit — both training and inference use this copy:
python -m mlx_lm.convert --model ~/models/Ling-3.0-tiny \
    --q-bits 4 --mlx-path ~/models/Ling-3.0-tiny-4bit --trust-remote-code
```

> If quantization reports `StrictDataclassFieldValidationError: Field
> 'kda_lower_bound' expected float`, change `-5` / `0` to `-5.0` / `0.0` in
> `config.json` (see README Troubleshooting #1).

<a id="data"></a>

### Step 2: Generate the training dataset (programmatic synthesis)

```{code-cell}
%%bash
python data/synthesize.py   # fixed seed; writes train/valid/test.jsonl (1212/76/96)
```

Synthesis: random word pairs → roles (1 undercover / 5 civilians) → templated
speeches per round (undercover stays vague, civilians use discriminative
attributes) → rule-based vote labels. Vote samples are oversampled 3× to balance
task difficulty (train 1212 = 432 speak + 780 vote); valid/test keep the natural
distribution.

<a id="train"></a>

### Step 3: QLoRA fine-tuning

```{code-cell}
%%bash
cd train && bash train.sh    # 1500 steps ≈ 2.5h on M4 16GB; adapters land in adapters/
# equivalent: python -m mlx_lm lora -c train/lora_config.yaml
```

Key settings: 4-bit base + LoRA rank 8 on the last 16/24 layers, batch 1 ×
grad-accum 4, lr 1e-4, `mask_prompt`, gradient checkpointing.

**Segmented training on 16G** (continuous runs die at ~625 steps with Metal OOM):

```bash
python -m mlx_lm lora ... --iters 600 --adapter-path ../adapters-seg1
python -m mlx_lm lora ... --iters 300 \
    --resume-adapter-file ../adapters-seg1/0000600_adapters.safetensors \
    --adapter-path ../adapters-seg2
```

Checkpoint filenames are **7-digit** zero-padded (`0000600_adapters.safetensors`);
optimizer state resets between segments and the LR re-warms, so val-loss spikes at
segment starts are expected.

<a id="eval"></a>

### Step 4: Before/after evaluation (incl. self-play)

```{code-cell}
%%bash
# 96-item test set (6 held-out word pairs), seeded sampling temp=0.7 top_p=0.9 seed=42
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit \
    --data data/test.jsonl --out eval/base.json
python eval/evaluate.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters \
    --data data/test.jsonl --out eval/lora.json

# Self-play win rate: 6 model seats + scripted referee (20 games; 200-game random baseline)
python eval/selfplay.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters \
    --games 20 --out eval/selfplay.json
python eval/selfplay.py --model ~/models/Ling-3.0-tiny-4bit --policy random \
    --games 200 --out eval/selfplay-random.json
```

See "7. Typical Results" below for the comparison tables; per-round metrics are
recorded in `eval/metrics-history.jsonl`.

### Step 5: Interactive demo

```{code-cell}
%%bash
python infer/chat.py --model ~/models/Ling-3.0-tiny-4bit --adapter-path adapters
```

<a id="flywheel"></a>

### Step 6: Data flywheel (optional, advanced)

```{code-cell}
%%bash
# After every ≥5 new live games:
python flywheel.py build --out data/live-vN.jsonl          # game state -> samples (truth labels)
python flywheel.py merge --live data/live-vN.jsonl data/api-distill-vN.jsonl \
    --out data/train-vN.jsonl                              # merge increments (≤30% cap)
python data/rerender.py data/live-vN.jsonl data/api-distill-vN.jsonl  # lossless re-render on format change
```

Three sampling disciplines: ground-truth vote labels are the core increment
(DAgger-style); self-imitation (speak) is rate-limited; games touching held-out
word pairs are skipped. **Adoption gate** (anti-collapse): `vote_format_ok = 1.0`,
`speak_format_ok` no drop vs previous round, leak rate up by ≤2pt — only a gate
passing adapter goes live. See README "Data flywheel".

<a id="apply"></a>

### Step 7: Join the Avernet game table (application side)

+++

Serve the model via `mlx_lm.server` (single instance, dual role: base/LoRA
switched per request through the `adapters` field); the player bot joins the
table over the BCN V3 WebSocket protocol in pure Python. Integration code and the
ops playbook live in Avernet
[PR #3970](https://github.com/inclusionAI/Avernet/pull/3970) and
[integration verification](eval/verification/INTEGRATION.md); inference weights are on
[Release `undercover-player-adapter-v2`](https://github.com/lplalbert/ling-cookbook/releases/tag/undercover-player-adapter-v2).

+++

<a id="results"></a>

## 7. Typical Results
> Historical results: these numbers use the previous parsing/rendering protocol. They do not validate the current default thinking-on protocol; current strict re-evaluation is recorded in [the verification report](eval/verification/REPORT.md).


test.jsonl (96 items, 6 held-out word pairs), seeded sampling
(temp=0.7, top_p=0.9, seed=42). Three-way comparison on the windowed test
(48 items trigger history folding):

| Metric | Base (4-bit) | v2 adapter (adopted) | v3 adapter (windowed retrain) |
|---|---|---|---|
| Length compliance (≤25 chars) | 93.3% | **100%** | 88.3% |
| Naturalness (no think-prefix) | 8.3% | 100% | 100% |
| Speech format compliance | 3.3% | **93.3%** | 78.3% |
| Word-leak rate (lower is better) | 5.0% | **6.7%** | 11.7% |
| Vote format validity | 52.8% | **100%** | **100%** |
| Rule-label match (observed) | 11.1% | 25.0% | 19.4% |

Gate verdict: v3 did not pass (speak_format 0.783, leak +5pt), so **v2 remains
adopted** — the first time the gate actually blocked a regression. v2 performed similarly on folded and unfolded subsets (folded/unfolded speak_format 0.939/0.926), so windowed
inference with v2 passed the historical gate.

The self-play command above is optional; no current self-play win-rate improvement is claimed. The three-seed validation and its uncertainty are recorded in [the verification report](eval/verification/REPORT.md).

+++

## 8. FAQ & Future Directions

- **Raw output starts with `</think>`**: closing an empty thinking block is a valid training target, not evidence of an offset bug. Use `thinking_option="on"` consistently and extract visible text via `clean_output()`.
- **Greedy-decoding collapse** (all-abstain votes / canned speeches): seeded
  sampling plus a runtime cleaning layer.
- **Rule-label matching limitations**: rule-based labels and missing observable state need further investigation (3-step
  attribution in README); next steps = flywheel truth votes vs teacher-distilled
  votes (see the README ablation).
- **Train/inference distribution match**: any rule that folds/truncates input at
  inference time must be covered across its full value range in training data
  (README Troubleshooting #9).
- Full pitfall log (9 entries) in [README.md](README.md) Troubleshooting.

+++

## 9. Acknowledgements

- Thanks to the Ant open-source camp and the maintainers of
  `inclusionAI/ling-cookbook` and `inclusionAI/Avernet`;
- Game-rule semantics follow Avernet's `undercover-game-referee`;
- Code, data synthesis, training configs, evaluation and docs were produced with
  [Claude Code](https://claude.com/claude-code) under human supervision
  (disclosure: [AI_USAGE.md](AI_USAGE.md)).


## Final verification / 最终核验

The complete validation retains v2: 12 jobs, three decoding seeds, and 1,056 recorded outputs. The candidate hit 11/48 repeated civilian votes versus 10/48 for v2; only 16 independent states were tested. No stable decision-quality improvement was established. See [verification report](eval/verification/REPORT.md) and [integration measurements](eval/verification/INTEGRATION.md).
