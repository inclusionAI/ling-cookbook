"""Validation-only seat renumbering diagnostic; no training or adoption."""
import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data'))
from prompting import render_player_state


def permute_seats(sample, mapping):
    result = copy.deepcopy(sample)
    old = sample['meta']
    count = old['n_players']
    expected = set(range(1, count + 1))
    if set(mapping) != expected or set(mapping.values()) != expected:
        raise ValueError('Seat mapping must be a bijection')
    meta = result['meta']
    meta['my_seat'] = mapping[old['my_seat']]
    meta['uc_seat'] = mapping[old['uc_seat']]
    meta['alive_seats'] = sorted(mapping[s] for s in old['alive_seats'])
    prompt = result['messages'][0]['content']
    before = render_player_state(old['my_seat'], old['alive_seats'])
    after = render_player_state(meta['my_seat'], meta['alive_seats'])
    if prompt.count(before) != 1:
        raise ValueError('State block missing or ambiguous')
    prompt = prompt.replace(before, after, 1)
    prompt = re.sub(r'^(\d+)号：', lambda m: f'{mapping[int(m[1])]}号：', prompt, flags=re.M)
    result['messages'][0]['content'] = prompt
    target = result['messages'][1]['content']
    match = re.fullmatch(r'我投(\d+)号', target)
    if match:
        target = f'我投{mapping[int(match[1])]}号'
    result['messages'][1]['content'] = target
    meta['target'] = target
    if meta.get('label_seat') is not None:
        meta['label_seat'] = mapping[old['label_seat']]
    return result


def reverse_seats(sample):
    count = sample['meta']['n_players']
    return permute_seats(sample, {seat: count + 1 - seat for seat in range(1, count + 1)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-path', type=Path, default=ROOT / 'adapters-state-pilot-state')
    parser.add_argument('--out', type=Path, default=ROOT / 'eval/seat-permutation-probe')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    samples = []
    for index, line in enumerate((ROOT / 'data-v5-state/valid.jsonl').open()):
        sample = json.loads(line)
        if sample['task'] == 'vote' and sample['role'] == '平民':
            sample['meta']['eval_seed'] = 42 + index
            samples.append(sample)
    reversed_samples = [reverse_seats(s) for s in samples]
    for original, reversed_sample in zip(samples, reversed_samples):
        restored = reverse_seats(reversed_sample)
        if restored['messages'] != original['messages']:
            raise ValueError('Seat mapping is not reversible')
    args.out.mkdir(parents=True)
    status = {'status': 'running', 'completed': [], 'samples': len(samples),
              'limitations': ['One seed and a small validation subset.',
                              'Historical utterance order is preserved; only speaker identities change.',
                              'This is an equivariance diagnostic, not a proof of reasoning ability.']}

    def save():
        (args.out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        for name, rows in [('original', samples), ('reversed', reversed_samples)]:
            data = args.out / f'{name}.jsonl'
            data.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            existing = ROOT / 'eval/state-v2-training-pilot/valid-state.json'
            if name == 'original' and existing.exists():
                report = json.loads(existing.read_text())
                summary = report['summary']
                current_hash = hashlib.sha256((ROOT / 'data-v5-state/valid.jsonl').read_bytes()).hexdigest()
                if (Path(summary['adapter_path']).resolve() == args.adapter_path.resolve()
                        and summary.get('sample_seeds') is True
                        and summary.get('data_sha256') == current_hash
                        and summary.get('decoding') == 'temp=0.7,top_p=0.9,seed=42'):
                    records = [report['records'][row['meta']['eval_seed'] - 42] for row in rows]
                    if any(record['target'] != row['messages'][1]['content'] for record, row in zip(records, rows)):
                        raise ValueError('Reusable report order mismatch')
                    (args.out / 'original.json').write_text(json.dumps(
                        {'reused_from': str(existing), 'records': records}, ensure_ascii=False, indent=2) + '\n')
                    status['completed'].append(name)
                    status['original_reused'] = True
                    save()
                    continue
            command = [sys.executable, str(ROOT / 'eval/evaluate.py'), '--model',
                       str(Path.home() / 'models/Ling-3.0-tiny-4bit'), '--adapter-path',
                       str(args.adapter_path.resolve()), '--data', str(data.resolve()), '--seed', '42',
                       '--sample-seeds', '--max-tokens', '256', '--out', str((args.out / f'{name}.json').resolve())]
            status['current'] = name
            save()
            with (args.out / f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            status['completed'].append(name)
            save()
        reports = {name: json.loads((args.out / f'{name}.json').read_text()) for name in ['original', 'reversed']}
        details = []
        for sample, a, b in zip(samples, reports['original']['records'], reports['reversed']['records']):
            match = re.fullmatch(r'我投(\d+)号', a['output'])
            expected = (f"我投{sample['meta']['n_players'] + 1 - int(match[1])}号"
                        if match and 1 <= int(match[1]) <= sample['meta']['n_players'] else None)
            if a['output'] == '我弃权':
                expected = '我弃权'
            details.append({'id': sample['id'], 'original_output': a['output'],
                            'reversed_output': b['output'], 'mapped_original_output': expected,
                            'equivariant': a['format_ok'] and b['format_ok'] and b['output'] == expected})
        status.update(status='complete', current=None, details=details,
                      equivariant_count=sum(row['equivariant'] for row in details))
    except Exception as error:
        status.update(status='failed', error=str(error))
        save()
        raise
    save()


if __name__ == '__main__':
    main()
