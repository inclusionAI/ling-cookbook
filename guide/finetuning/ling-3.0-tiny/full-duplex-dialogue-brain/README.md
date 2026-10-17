<p align="right">
  <b>English</b> | <a href="README-zh.md">简体中文</a>
</p>

# Full-Duplex Dialogue Brain: LoRA Fine-tuning Ling-3.0-tiny for Turn-Taking

**Author:** [Dayuxiaoshui](https://github.com/Dayuxiaoshui)

A full-duplex voice assistant listens and talks at the same time. Besides *what* to say, it must constantly decide *when* to act: wait while the user pauses mid-sentence, give a short "mm-hmm", take the floor when the user is done, stop when interrupted, and ignore speech that is not meant for it.

This recipe fine-tunes [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) (7.9B total / 1.3B active MoE) with LoRA into such a **dialogue brain**. At every decision moment it reads the live ASR text and outputs one action tag, followed by a short reply when it takes the floor:

| Action | When |
| :--- | :--- |
| `[LISTEN]` | The user has not finished (mid-sentence pause, hesitation, trailing "and…"). Stay silent. |
| `[BACKCHANNEL]` | The user is in a long turn. Give a short acknowledgement without taking the floor. |
| `[SPEAK]` | The user finished a turn addressed to the assistant. Reply. |
| `[STOP]` | The assistant is speaking and the user interrupts with a new or changed request. Stop and respond. |
| `[IGNORE]` | The speech needs no response: a brief "yeah", the user talking to someone else, background speech. |

Evaluated on decision points built from [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1.0 / v1.5, the best checkpoint raises **accuracy from 0.581 to 0.885** and **macro F1 from 0.374 to 0.737**. It also cuts the rate of answering speech that should be ignored from **33.6% to 0.7%** and raises the interruption stop rate from **62.5% to 96.5%**. No Full-Duplex-Bench data is used for training.

---

## Contents

- [Task Formulation](#task-formulation)
- [Directory Layout](#directory-layout)
- [Environment](#environment)
- [Reproduction](#reproduction)
- [Results](#results)
- [Analysis](#analysis)
- [Limitations](#limitations)
- [Weights and Data](#weights-and-data)
- [AI Usage Report](#ai-usage-report)

---

## Task Formulation

One request = one decision moment. The input is the assistant state, what the user has said so far (ASR text), and the silence since the user's last word:

```text
Assistant state: speaking (currently answering: "Can you help me plan a weekend trip to Kyoto?")
New user speech: "wait, actually make it Osaka instead"
Silence since the last word: 0.3s
Action:
```

The output is an action tag, plus a reply after `[SPEAK]` / `[STOP]` (or the acknowledgement after `[BACKCHANNEL]`):

```text
[STOP] No problem, Osaka is a great choice too.
```

The system prompt and format live in [`fdb_brain/prompt.py`](fdb_brain/prompt.py) and are shared by data building, evaluation and inference. Because the action comes first, a downstream controller can act on the first few tokens without waiting for the reply to finish.

**Silence is fixed** in both training and evaluation: 0.8s when the assistant is silent, 0.3s while it is speaking. The label can therefore not be read off the pause length. Telling a mid-sentence pause from a finished turn has to come from *what the user said*, which is the part a text-only LLM can contribute on top of a VAD.

## Directory Layout

```text
full-duplex-dialogue-brain/
├── fdb_brain/prompt.py        # shared system prompt, input rendering, action parser
├── data/
│   ├── build_fdb_eval.py      # Full-Duplex-Bench annotations -> 1862 eval decision points
│   ├── synth_scenarios.py     # generate dialogue scenarios with the base model
│   └── build_train.py         # scenarios -> labelled, pre-tokenized SFT rows
├── train/
│   ├── train_lora.sh          # AReno LoRA SFT (training config)
│   └── sft_loader.py          # dataset loader passed to `areno train`
├── scripts/serve.sh           # start an OpenAI-compatible `areno serve` (optionally with LoRA)
├── eval/run_eval.py           # query + score; writes per-point jsonl and a report json
├── infer.py                   # inference example (single decision or scripted call)
└── results/*.report.json      # evaluation reports of every run in the tables below
```

## Environment

| Item | Version used |
| :--- | :--- |
| GPU | 8× NVIDIA H20 96GB (training uses all 8; serving uses 4) |
| Training / serving framework | AReno at commit `3cfa480` (`areno train`, `areno serve`) |
| Base model | `inclusionAI/Ling-3.0-tiny` |
| Eval data | Full-Duplex-Bench v1.0 + v1.5 (annotations only, no audio) |
| Python deps for data/eval scripts | `httpx`, `transformers` (tokenizer only) |

Set `ARENO_VENV=/path/to/venv` if `areno` is not on your `PATH`. Every step below runs from this directory.

```bash
export M=/path/to/Ling-3.0-tiny        # local snapshot of inclusionAI/Ling-3.0-tiny
export FDB=/path/to/Full-Duplex-Bench  # directory containing v1.0/ and v1.5/
```

## Reproduction

### 1. Build the evaluation set (Full-Duplex-Bench → decision points)

```bash
python data/build_fdb_eval.py --fdb-root $FDB --out data/fdb_eval.jsonl
```

Each FDB clip is converted into one or more text-level decision points:

| FDB subset | Decision point | Gold | n |
| :--- | :--- | :--- | ---: |
| v1.0 `candor_pause_handling` | text up to each annotated `[PAUSE]` | LISTEN | 277 |
| v1.0 `synthetic_pause_handling` | text up to each pause / full turn | LISTEN / SPEAK | 137 / 137 |
| v1.0 `icc_backchannel` | text up to each ≥0.3s gap in a long monologue | BACKCHANNEL (LISTEN also accepted) | 315 |
| v1.5 `user_interruption` | request (silent) / interruption (speaking) | SPEAK / STOP | 200 / 200 |
| v1.5 `user_backchannel` | request / "yeah"-style backchannel | SPEAK / IGNORE | 98 / 98 |
| v1.5 `talking_to_other` | request / the user talking to someone else | SPEAK / IGNORE | 100 / 100 |
| v1.5 `background_speech` | request / background speech | SPEAK / IGNORE | 100 / 100 |

Total: **1862** points.

### 2. Generate training scenarios with the base model

The training data is fully synthetic. Serve the **base** model with thinking off, then ask it to write dialogue scenarios. Each scenario covers one topic × persona and contains a request and reply, an interruption and its reply, side talk, background speech, a long story, a topic switch, and more (14 fields, see `GEN_PROMPT` in [`data/synth_scenarios.py`](data/synth_scenarios.py)).

```bash
bash scripts/serve.sh $M 0,1,2,3 8100 /tmp/serve-base.log --disable-thinking
python data/synth_scenarios.py --base-url http://127.0.0.1:8100 --n 1500 --seed 1 --out data/scenarios_r2.jsonl
python data/synth_scenarios.py --base-url http://127.0.0.1:8100 --n 800  --seed 2 --out data/scenarios_r2b.jsonl
```

Malformed generations are dropped (1500 → 984 and 800 → 498 kept). The released model also used 1200 scenarios (`data/scenarios.jsonl`, seed 0) from an earlier 8-field version of the prompt. These are included in the released dataset, so you can download them instead of regenerating.

### 3. Expand scenarios into labelled decision points

```bash
python data/build_train.py \
  --scenarios data/scenarios.jsonl data/scenarios_r2.jsonl data/scenarios_r2b.jsonl \
  --fdb-eval data/fdb_eval.jsonl --tokenizer $M --out-dir data/
```

Each scenario expands into up to 18 decision points covering all five actions. Examples: a request cut mid-sentence → LISTEN; the full request → SPEAK + reply; an interruption while speaking → STOP + reply; side talk while speaking → IGNORE; a story cut after a full sentence → BACKCHANNEL. The full list is in the docstring of [`data/build_train.py`](data/build_train.py). Three details matter:

- **ASR-style noise.** User text is randomly rendered without punctuation and with `[UM]` / `[UH]` fillers (35% each), so the model cannot rely on a final `?` or `.`.
- **Mid-turn hard negatives.** Each story gives 2 windows that end on a complete-sounding clause before the real end (sometimes starting mid-sentence) → LISTEN. Without these, the model learns "complete sentence ⇒ SPEAK" and takes over in pauses (see r2 → r3 below).
- **No eval leakage.** Any scenario whose text overlaps an FDB eval text is dropped.

Rows are written **pre-tokenized** (`tokens` + `prompt_mask`) with the chat template that `areno serve --disable-thinking` renders, so training and serving see identical prompts. This is needed because AReno's raw-text SFT path does not recognise Ling's `<role>` markers and would wrap the prompt twice.

Output: `train.jsonl` with 32,423 rows (LISTEN 12,521 / IGNORE 7,635 / SPEAK 7,192 / STOP 3,951 / BACKCHANNEL 1,124) and `val.jsonl` with 1,662 rows.

### 4. LoRA SFT

```bash
EPOCHS=2 TP=4 SAVE_INTERVAL=200 bash train/train_lora.sh $M data/train.jsonl ckpt/lora-r3 0,1,2,3,4,5,6,7
```

Training config ([`train/train_lora.sh`](train/train_lora.sh)):

| Hyper-parameter | Value |
| :--- | :--- |
| LoRA rank / alpha | 32 / 64 (q, k, v, o, gate, up, down projections) |
| Batch size / micro batch | 64 / 16 |
| Epochs | 2 (1014 steps) |
| LR schedule | 2e-4 → 2e-5, cosine |
| Max prompt / response tokens | 512 / 96 |
| Parallelism | TP=4 × DP=2 on 8× H20, ≈4.7 s/step (≈80 min total) |

The loss is computed on the response only (`[ACTION] reply`). The final loss is ≈0.32. AReno's SFT trainer saves only when `(step + 1) % save_interval == 0`. `SAVE_INTERVAL=200` keeps `step_000200 … step_001000` for checkpoint selection. **The released adapter is `step_000800`.**

### 5. Serve and evaluate

```bash
# baseline
bash scripts/serve.sh $M 0,1,2,3 8100 /tmp/serve-base.log --disable-thinking
python eval/run_eval.py --base-url http://127.0.0.1:8100 --data data/fdb_eval.jsonl --out results/baseline_nothink.jsonl

# fine-tuned (pass the LoRA rank explicitly: areno serve defaults to 8)
bash scripts/serve.sh $M 4,5,6,7 8101 /tmp/serve-lora.log --disable-thinking \
  --lora-rank 32 --lora-alpha 64 --lora-adapter-path ckpt/lora-r3/step_000800
python eval/run_eval.py --base-url http://127.0.0.1:8101 --data data/fdb_eval.jsonl --out results/lora_r3_step_000800.jsonl
```

`run_eval.py` uses greedy decoding (temperature 0, max 128 tokens). It writes every prediction to `*.jsonl` and the metrics to `*.report.json`. Re-run with `--score-only` to re-score an existing jsonl without querying.

### 6. Inference example

```bash
python infer.py --base-url http://127.0.0.1:8101 --demo
```

Real output of the released adapter (`step_000800`):

```text
[silent  ] user: 'Hey, can you help me plan a'
           -> LISTEN
[silent  ] user: 'Hey, can you help me plan a weekend trip to Kyoto?'
           -> SPEAK  "Sure, I can help you plan a trip to Kyoto. Let's start by picking the dates and your budget."
[speaking] user: 'yeah'
           -> IGNORE
[speaking] user: 'honey, can you grab my keys?'
           -> IGNORE
[speaking] user: 'wait, actually make it Osaka instead'
           -> STOP  'No problem, Osaka is a great choice too.'
[silent  ] user: 'So last time I went to Japan I got completely lost in the subway, and then'
           -> LISTEN
[silent  ] user: 'So last time I went to Japan I got completely lost in the subway, and then a stranger walked me all the way to my hotel.'
           -> SPEAK  "That's a great story, I bet you had a lot of fun getting back on the right train."
```

For a single decision, use `python infer.py --agent-state speaking --history "..." --user "..." --silence 0.3`. Any OpenAI-compatible server that applies Ling's chat template with thinking disabled and loads the PEFT adapter works the same way.

## Results

### Metrics

| Metric | Definition | Better |
| :--- | :--- | :---: |
| accuracy | prediction ∈ accepted labels | ↑ |
| macro F1 | mean F1 over the 5 actions (strict: BACKCHANNEL must be predicted as BACKCHANNEL) | ↑ |
| takeover rate (mid-turn) | gold LISTEN/BACKCHANNEL, predicted SPEAK/STOP: the assistant cuts in | ↓ |
| false response rate | gold IGNORE, predicted SPEAK/STOP: the assistant answers speech not meant for it | ↓ |
| respond rate (end of turn) | gold SPEAK, predicted SPEAK/STOP | ↑ |
| stop rate (interruption) | gold STOP, predicted STOP | ↑ |
| latency | mean wall time per request (action + reply), 16 concurrent requests on 4× H20 | ↓ |

### Baseline vs. fine-tuned (Full-Duplex-Bench, 1862 points)

| Metric | Base (thinking off) | Base (thinking on) | LoRA r1 | LoRA r2 | **LoRA r3 @800 (released)** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| accuracy ↑ | 0.581 | 0.445 | 0.806 | 0.857 | **0.885** |
| macro F1 ↑ | 0.374 | 0.380 | 0.707 | **0.810** | 0.737 |
| takeover rate (mid-turn) ↓ | **0.023** | 0.088 | 0.089 | 0.155 | 0.081 |
| false response rate ↓ | 0.336 | 0.530 | **0.000** | 0.007 | 0.007 |
| respond rate (end of turn) ↑ | 0.394 | 0.301 | 0.735 | **0.811** | 0.770 |
| stop rate (interruption) ↑ | 0.625 | 0.430 | 0.530 | **0.980** | 0.965 |
| parse failure ↓ | 0.001 | 0.343 | 0 | 0 | 0 |
| mean output tokens | 39.5 | 172.4 | 10.7 | 11.5 | 11.4 |
| latency (s) ↓ | 1.92 | 3.49 | **1.20** | 1.43 | 1.34 |

Runs:

- **r1:** 1,200 scenarios, 8 fields, 3 epochs.
- **r2:** adds 1,482 scenarios with topic switches, statements and short stories, plus ASR-style noise.
- **r3:** adds mid-turn hard negatives, 2 epochs, checkpoint every 200 steps.

### Per-source accuracy

| Source (gold) | n | Base | r1 | r2 | r3 @800 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| candor_pause_handling (LISTEN) | 277 | 0.953 | 0.913 | 0.722 | 0.924 |
| synthetic_pause_handling (LISTEN) | 137 | 0.964 | 0.985 | 0.964 | 0.978 |
| synthetic_pause_handling (SPEAK) | 137 | 0.051 | 0.190 | 0.533 | 0.453 |
| icc_backchannel (BACKCHANNEL / LISTEN) | 315 | 0.984 | 0.768 | 0.813 | 0.886 |
| v1.5 user_interruption (STOP) | 200 | 0.625 | 0.530 | 0.980 | 0.965 |
| v1.5 user_backchannel (IGNORE) | 98 | 0.010 | 1.000 | 1.000 | 1.000 |
| v1.5 talking_to_other (IGNORE) | 100 | 0.000 | 1.000 | 1.000 | 1.000 |
| v1.5 background_speech (IGNORE) | 100 | 0.000 | 1.000 | 0.980 | 0.980 |
| v1.5 requests, all subsets (SPEAK) | 498 | 0.43–0.67 | 0.74–0.99 | 0.76–1.00 | 0.76–0.99 |

### r3 checkpoint sweep

| Metric | @200 | @400 | @600 | **@800** | @1000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 0.840 | 0.850 | 0.873 | **0.885** | 0.882 |
| macro F1 | 0.709 | **0.796** | 0.731 | 0.737 | 0.736 |
| takeover rate ↓ | 0.218 | 0.115 | **0.053** | 0.081 | 0.060 |
| respond rate ↑ | **0.833** | 0.753 | 0.701 | 0.770 | 0.735 |
| stop rate ↑ | 0.875 | **1.000** | 0.985 | 0.965 | 0.985 |

All reports are in [`results/`](results/).

## Analysis

- **The base model cannot hold back.** With thinking off it answers almost everything addressed at it, including side talk and background speech: false response rate 33.6%, IGNORE recall 0.3%. It also stays silent on most finished turns (respond rate 39.4%). Its high CANDOR / ICC accuracy comes from defaulting to LISTEN when the assistant is silent, not from understanding the turn.
- **Thinking makes it worse.** It is slower (3.5s, 172 tokens), and 34% of outputs never reach an action tag within 128 tokens. A turn-taking decision has to be fast, so this recipe trains and serves with thinking disabled.
- **Fine-tuning solves "is this for me?"** Every run reaches ≈100% on the three IGNORE subsets. r2 and r3@800 stop on ≥96.5% of interruptions. Outputs shrink to ≈11 tokens and latency drops by 25–37%.
- **The hard part is end-of-turn vs. mid-turn pause, and the two pull against each other.** r2 learned to respond to finished turns (respond rate 0.811) but started cutting into CANDOR pauses (LISTEN 0.722, takeover 0.155). The r3 hard negatives (complete-sounding clauses before the real end → LISTEN) bring CANDOR back to 0.924 and halve the takeover rate, at a small cost in respond rate. The checkpoint sweep exposes the same trade-off: @600 has the lowest takeover (0.053), @200 the highest respond rate (0.833). We release @800 as the best overall accuracy with balanced takeover / respond rates. Pick @400 or r2 if you value macro F1 and responsiveness over not interrupting.
- **Why macro F1 drops from r2 to r3.** r3@800 never outputs BACKCHANNEL on ICC: it listens instead (279 / 315), which is accepted for accuracy but counts as a miss in strict macro F1. BACKCHANNEL is only 3.5% of the training data, and the hard negatives teach "keep listening" on very similar inputs. Most of the remaining errors are finished `synthetic_pause_handling` turns predicted as LISTEN (129 SPEAK → LISTEN).

## Limitations

- **Text only, fixed silence.** The model sees ASR text and a silence value, but both training and evaluation use a fixed silence. It has not learned to use pause length, prosody or audio. In a real system it should sit behind a VAD / endpointer that decides *when* to ask it.
- **Text-level proxy of Full-Duplex-Bench.** The numbers come from decision points derived from FDB annotations. They are not FDB's official audio-level metrics and should not be compared directly with leaderboard results from speech-to-speech models.
- **Checkpoint selection used the eval set.** The r3 checkpoint was picked by FDB accuracy, so the r3@800 numbers are mildly optimistic. r1 / r2 use their final checkpoints.
- **Synthetic, English-only training data** generated by the base model itself. Topic and phrasing diversity is limited, for example in how users open a topic switch.
- **BACKCHANNEL is under-produced** (see Analysis). If you need active backchannels, raise `P_BACKCHANNEL` in `build_train.py` or use r2.

## Weights and Data

| Artifact | Link |
| :--- | :--- |
| LoRA adapter (r3 `step_000800`, PEFT, ≈2.4 GB) | [Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-LoRA](https://huggingface.co/Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-LoRA) |
| Training data (scenarios + `train.jsonl` / `val.jsonl`) | [Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT](https://huggingface.co/datasets/Dayuxiaoshui/Ling-3.0-tiny-Full-Duplex-Brain-SFT) |
| Base model | [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |
| Evaluation source | [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench); rebuild `fdb_eval.jsonl` with step 1 |

## AI Usage Report

The whole project was carried out together with an AI coding agent (Claude Code), running on the training server over SSH. The agent wrote most of the code in this directory: the FDB-to-decision-point converter, scenario generator, data expander, training / serving wrappers and evaluation script. It also launched and monitored the training and evaluation runs and drafted this document. The human side defined the task, picked which direction to pursue after each round (more data in r2; hard negatives plus a checkpoint sweep in r3), and checked the results. The most useful contributions were the ones grounded in reading code and data rather than in general knowledge:

- It found by reading AReno's source that the raw-text SFT path does not recognise Ling's `<role>` chat markers, which led to the pre-tokenized data format.
- It noticed that the SFT trainer has no final save.
- It read the r2 confusion matrix (CANDOR pauses → SPEAK) and proposed mid-turn hard negatives as the fix.
- It proposed the checkpoint sweep to make the takeover / respond trade-off visible.
- It proposed fixing the silence value so the model cannot shortcut on pause length.

Several suggestions were unreliable and had to be caught:

- **Prompt examples were copied.** The first generator prompt included literal example utterances, and the base model copied them verbatim into many scenarios. The examples were removed, and the prompt now asks for varied openings.
- **The agent broke its own processes.** It killed its own SSH session twice with an over-broad `pkill -f` pattern. An earlier session hung on a 20-minute `sleep` and stopped responding.
- **Accuracy alone was misleading.** Accuracy-only reporting would have hidden that r3 never emits BACKCHANNEL. Strict macro F1 and the confusion matrix had to be added to see it.
- **Checkpoint selection used the eval set.** Picking the r3 checkpoint by FDB accuracy is a form of tuning on the test set. We report it in [Limitations](#limitations) instead of hiding it.

Our takeaway: the agent speeds up the engineering loop a lot. Every claim about model quality still has to be checked against per-source numbers and the raw predictions.
