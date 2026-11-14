"""Complete the vote pilot audit in one run and write a final decision report.

Three decoding seeds are repetitions on the same validation cases, not independent
training runs. This script never trains, publishes, or replaces an adapter.
"""
import copy
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'data')]
from evaluate import eval_sample
from seat_probe import reverse_seats

SEEDS = [7, 42, 123]
CONDITIONS = [
    ('deployed-v2', 'adapters-v2-run3', 'data/valid.jsonl', False),
    ('v2-state', 'adapters-v2-run3', 'data-v5-state/valid.jsonl', True),
    ('repeat-control', 'adapters-vote-permutation-pilot-control', 'data-v5-state/valid.jsonl', True),
    ('permuted', 'adapters-vote-permutation-pilot-state', 'data-v5-state/valid.jsonl', True),
]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def metrics(records):
    speech = [row for row in records if row['task'] == 'speak']
    votes = [row for row in records if row['task'] == 'vote']
    civilian = [row for row in votes if row.get('role') == '平民']
    def rate(rows, field):
        return sum(bool(row[field]) for row in rows) / len(rows) if rows else None
    return {'n_speak': len(speech), 'n_vote': len(votes), 'n_civilian_vote': len(civilian),
            'speech_format': rate(speech, 'format_ok'), 'speech_leak': rate(speech, 'leak'),
            'vote_format': rate(votes, 'format_ok'), 'civilian_hit': rate(civilian, 'undercover_hit'),
            'label_match': rate(votes, 'correct')}


def bootstrap_difference(jobs, treatment, control):
    by_condition = {}
    for condition in [treatment, control]:
        values = {}
        for seed in SEEDS:
            for row in jobs[f'{condition}-{seed}']['records']:
                if row['task'] == 'vote' and row.get('role') == '平民':
                    values.setdefault(row['id'], []).append(float(row['undercover_hit']))
        by_condition[condition] = {key: statistics.mean(value) for key, value in values.items()}
    keys = sorted(by_condition[treatment])
    if set(keys) != set(by_condition[control]):
        raise ValueError('Paired case IDs differ')
    differences = [by_condition[treatment][key] - by_condition[control][key] for key in keys]
    rng = random.Random(2026)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
    return {'independent_validation_cases': len(keys), 'mean_difference': statistics.mean(differences),
            'case_bootstrap_95_percent_interval': [samples[124], samples[4874]],
            'note': 'Resampling cases after averaging decoding seeds; no independent training-seed uncertainty.'}


def main():
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    lock = (ROOT / 'eval/.audit-queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = ROOT / 'eval/final-vote-audit'
    output.mkdir(exist_ok=True)
    dependencies = [ROOT / 'eval/final_vote_audit.py', ROOT / 'eval/evaluate.py',
                    ROOT / 'eval/seat_probe.py', ROOT / 'data/prompting.py']
    for _, adapter, data, _ in CONDITIONS:
        dependencies.extend([ROOT / data, ROOT / adapter / 'adapters.safetensors',
                             ROOT / adapter / 'adapter_config.json'])
    fingerprint = {str(path.relative_to(ROOT)): digest(path) for path in set(dependencies)}
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise RuntimeError('Dependencies changed; use a fresh experiment directory')
    manifest.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2) + '\n')
    status = {'status': 'running', 'total_jobs': len(CONDITIONS) * len(SEEDS),
              'completed': [], 'seeds': SEEDS, 'max_tokens': 256, 'thinking_option': 'on'}

    def save():
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output / 'status.json')

    jobs = {}
    save()
    try:
        for condition, adapter, data, probe in CONDITIONS:
            rows = [json.loads(line) for line in (ROOT / data).open()]
            model, tokenizer = load(str(Path.home() / 'models/Ling-3.0-tiny-4bit'), adapter_path=str(ROOT / adapter))
            for seed in SEEDS:
                name = f'{condition}-{seed}'
                result_path = output / f'{name}.json'
                if result_path.exists():
                    jobs[name] = json.loads(result_path.read_text())
                    status['completed'].append(name)
                    save()
                    continue
                status.update(current=name, phase='original', current_sample=0)
                save()
                records = []
                for index, row in enumerate(rows):
                    mx.random.seed(seed + index)
                    prompt = tokenizer.apply_chat_template(row['messages'][:-1], add_generation_prompt=True,
                                                          tokenize=False, thinking_option='on')
                    raw = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                                   sampler=make_sampler(temp=0.7, top_p=0.9), verbose=False)
                    scored = eval_sample(row, raw, require_think_end=True)
                    scored['id'] = row['id']
                    records.append(scored)
                    mx.clear_cache()
                    status['current_sample'] = index + 1
                    if (index + 1) % 10 == 0:
                        print(name, 'original', index + 1, '/', len(rows), flush=True)
                        save()
                reversed_records = []
                consistency = []
                if probe:
                    status.update(phase='reversed', current_sample=0)
                    save()
                    for index, row in enumerate(rows):
                        if row['task'] != 'vote' or row['role'] != '平民':
                            continue
                        reversed_row = reverse_seats(row)
                        mx.random.seed(seed + index)
                        prompt = tokenizer.apply_chat_template(reversed_row['messages'][:-1], add_generation_prompt=True,
                                                              tokenize=False, thinking_option='on')
                        raw = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                                       sampler=make_sampler(temp=0.7, top_p=0.9), verbose=False)
                        scored = eval_sample(reversed_row, raw, require_think_end=True)
                        scored['id'] = row['id']
                        reversed_records.append(scored)
                        original = records[index]
                        expected = original['output']
                        if expected.startswith('我投') and original['format_ok']:
                            seat = int(expected[2:-1])
                            expected = f"我投{row['meta']['n_players'] + 1 - seat}号"
                        consistency.append({'id': row['id'], 'original': original['output'],
                                            'reversed': scored['output'],
                                            'consistent': original['format_ok'] and scored['format_ok']
                                                          and expected == scored['output'],
                                            'original_abstained': original['output'] == '我弃权'})
                        mx.clear_cache()
                        status['current_sample'] = len(reversed_records)
                        save()
                result = {'condition': condition, 'seed': seed, 'metrics': metrics(records),
                          'records': records, 'reversed_records': reversed_records, 'consistency': consistency}
                temporary = output / f'{name}.tmp'
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                temporary.replace(result_path)
                jobs[name] = result
                status['completed'].append(name)
                save()
                print('DONE', name, flush=True)
            del model, tokenizer
            gc.collect()
            mx.clear_cache()

        aggregate = {}
        for condition, _, _, probe in CONDITIONS:
            reports = [jobs[f'{condition}-{seed}'] for seed in SEEDS]
            aggregate[condition] = {metric: statistics.mean(report['metrics'][metric] for report in reports)
                                    for metric in ['speech_format', 'speech_leak', 'vote_format', 'civilian_hit', 'label_match']}
            if probe:
                values = [item for report in reports for item in report['consistency']]
                nonabstain = [item for item in values if not item['original_abstained']]
                aggregate[condition]['renumber_consistency'] = sum(x['consistent'] for x in values) / len(values)
                aggregate[condition]['nonabstain_consistency'] = (sum(x['consistent'] for x in nonabstain) / len(nonabstain)
                                                                 if nonabstain else None)
        comparisons = {control: bootstrap_difference(jobs, 'permuted', control)
                       for control in ['repeat-control', 'v2-state', 'deployed-v2']}
        candidate = aggregate['permuted']
        deployed = aggregate['deployed-v2']
        gates = {'speech_format_nonregression': candidate['speech_format'] >= deployed['speech_format'] - 0.02,
                 'speech_leak_nonregression': candidate['speech_leak'] <= deployed['speech_leak'] + 0.02,
                 'vote_format_nonregression': candidate['vote_format'] >= deployed['vote_format'] - 0.02,
                 'civilian_hit_evidence': all(value['case_bootstrap_95_percent_interval'][0] > 0 for value in comparisons.values()),
                 'renumber_consistency_nonregression': candidate['renumber_consistency'] >= aggregate['repeat-control']['renumber_consistency']}
        conclusion = ('promising_but_requires_independent_training_seeds_and_heldout_confirmation'
                      if all(gates.values()) else 'retain_deployed_v2_do_not_adopt_candidate')
        final = {'aggregate': aggregate, 'comparisons': comparisons, 'gates': gates,
                 'decision': conclusion, 'automatic_adoption': False,
                 'limitations': ['Validation only; no new held-out test selection.',
                                 'Three decoding seeds, not three training seeds.',
                                 '16 independent civilian validation states; repeated seeds do not create new states.',
                                 'Thresholds finalized after the initial single-seed pilot; this is exploratory confirmation.']}
        (output / 'final-report.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n')
        lines = ['# Vote pilot final audit', '', '| Condition | Speech format | Leak | Vote format | Civilian hit |',
                 '|---|---:|---:|---:|---:|']
        for condition, values in aggregate.items():
            lines.append('| ' + condition + ' | ' + ' | '.join(f'{values[key]:.1%}' for key in
                         ['speech_format', 'speech_leak', 'vote_format', 'civilian_hit']) + ' |')
        lines.extend(['', 'Decision: ' + conclusion, '', 'No adapter was replaced.', '',
                      'Three decoding seeds on the same validation cases; not independent training runs.',
                      'See final-report.json for case-bootstrap intervals, gates, and limitations.'])
        (output / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
        status.update(status='complete', current=None, phase=None, decision=conclusion)
        save()
        print('FINAL', conclusion, flush=True)
    except Exception as error:
        status.update(status='failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
