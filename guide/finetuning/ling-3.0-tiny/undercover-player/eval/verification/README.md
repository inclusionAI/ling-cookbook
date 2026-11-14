# Reproduce the validation evidence

Run from the recipe directory after preparing the public 4-bit base model and downloading both assets from the v2 Release into `adapters-v2-run3/`.

```bash
python data/build_state_dataset.py --source data --out data-v5-state --control-out data-v5-control
python data/build_vote_permutations.py
python train/state_pilot.py --steps 60 --data-prefix data-v6-vote --experiment vote-permutation-training-pilot --adapter-prefix adapters-vote-permutation-pilot
python eval/final_vote_audit.py
```

The scripts refuse to overwrite an existing experiment. Use a fresh checkout/output directory to repeat it. No candidate is automatically adopted. Fixed initial adapter/data hashes and twelve result files accompany the summary; three decoding seeds are not three independently trained models.

- REPORT.md: final local model recommendation and uncertainty.
- INTEGRATION.md: independently measured local service behavior and its limits.
- final-report.json: aggregate metrics, exploratory intervals, and gates.
- `*-7.json`, `*-42.json`, `*-123.json`: scored original/reversed validation outputs.
- real-table-acceptance.json: three completed isolated real-BCS games with one MLX player and five scripted controls; not a mixed-LLM benchmark.
- release-integrity.json: public Release SHA256 compared with the starting adapter.

Historical test tables elsewhere use earlier parsing/rendering protocols; do not combine those denominators with this validation set. Real-table acceptance is recorded separately in the Avernet player PR; local HTTP tests do not establish game win rate.
