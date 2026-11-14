"""Matched-count civilian vote experiment: repeat versus renumber seats.

Training selection uses the source generator's attribute dictionary as an oracle.
Validation/test retain all civilian votes; they are not filtered by model outputs.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
from seat_probe import permute_seats
from word_pairs import WORD_PAIRS

PAIRS = {f"{p['civilian']}/{p['undercover']}": p for p in WORD_PAIRS}


def usable(sample):
    if sample['task'] != 'vote' or sample['role'] != '平民':
        return False
    pair = PAIRS[sample['pair_key']]
    scores = Counter({seat: 0 for seat in sample['meta']['alive_seats']})
    for seat, text in re.findall(r'^(\d+)号：(.*)$', sample['messages'][0]['content'], re.M):
        scores[int(seat)] += sum(attr in text for attr in set(pair['undercover_attrs']))
    maximum = max(scores.values())
    leaders = [seat for seat, score in scores.items() if score == maximum]
    return (maximum > 0 and len(leaders) == 1 and leaders[0] == sample['meta']['uc_seat']
            and sample['messages'][1]['content'] == f'我投{leaders[0]}号')


def main():
    outputs = {name: ROOT / f'data-v6-vote-{name}' for name in ['control', 'state']}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError('Refusing to overwrite a prior experiment')
    source = ROOT / 'data-v5-state'
    samples = [json.loads(line) for line in (source / 'train.jsonl').open()]
    selected, seen = [], set()
    for sample in samples:
        prompt = sample['messages'][0]['content']
        if usable(sample) and prompt not in seen:
            selected.append(sample)
            seen.add(prompt)
    rng = random.Random(42)
    training = {name: [] for name in outputs}
    for sample in selected:
        count = sample['meta']['n_players']
        permutations = [tuple(range(1, count + 1)), tuple(range(count, 0, -1))]
        while len(permutations) < 4:
            permutation = tuple(rng.sample(range(1, count + 1), count))
            if permutation not in permutations:
                permutations.append(permutation)
        for index, permutation in enumerate(permutations):
            mapping = dict(zip(range(1, count + 1), permutation))
            control = copy.deepcopy(sample)
            treatment = permute_seats(sample, mapping)
            for condition, row in [('control', control), ('state', treatment)]:
                row['id'] = f"{sample['id']}-view{index}"
                row['meta']['augmentation'] = condition
                training[condition].append(row)
    if not selected:
        raise ValueError('No usable training samples')
    manifest = {'unique_training_states': len(selected), 'views_per_state': 4,
                'selection': 'civilian, unique visible oracle evidence, source label agrees',
                'limits': ['Synthetic attribute oracle used for training selection.',
                           'Both conditions focus on votes and may regress on speech.',
                           'No automatic adoption.'], 'outputs': {}}
    heldout = {}
    for split in ['valid', 'test']:
        heldout[split] = []
        for index, line in enumerate((source / f'{split}.jsonl').open()):
            row = json.loads(line)
            if row['task'] == 'vote' and row['role'] == '平民':
                row['meta']['eval_seed'] = 42 + index
                heldout[split].append(row)
    train_pairs = {row['pair_key'] for row in selected}
    if train_pairs & {row['pair_key'] for row in heldout['test']}:
        raise ValueError('Held-out pair contamination')
    for condition, directory in outputs.items():
        directory.mkdir()
        for split, rows in [('train', training[condition]), *heldout.items()]:
            content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
            (directory / f'{split}.jsonl').write_text(content)
            manifest['outputs'][f'{condition}/{split}'] = {
                'rows': len(rows), 'sha256': hashlib.sha256(content.encode()).hexdigest()}
    for directory in outputs.values():
        (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
