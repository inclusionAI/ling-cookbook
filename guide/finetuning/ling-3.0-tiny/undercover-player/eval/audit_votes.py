"""Audit rule-label observability and paired vote errors without model calls.

The attribute dictionary is an oracle diagnostic, not a playable model policy.
"""
import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data'))
from word_pairs import WORD_PAIRS

PAIRS = {f"{p['civilian']}/{p['undercover']}": p for p in WORD_PAIRS}


def visible_rule_targets(sample):
    pair = PAIRS[sample['pair_key']]
    count = sample['meta']['n_players']
    utterances = re.findall(r'^(\d+)号：(.*)$', sample['messages'][0]['content'], re.M)
    attributes = set(pair['shared'] + pair['civilian_attrs'] + pair['undercover_attrs'])
    used = {seat: [] for seat in range(1, count + 1)}
    for seat, text in utterances:
        used[int(seat)].extend(attr for attr in attributes if attr in text)
    if sample['role'] == '平民':
        scores = {seat: sum(attr in pair['undercover_attrs'] for attr in attrs)
                  for seat, attrs in used.items()}
        maximum = max(scores.values())
        if maximum == 0:
            return {'我弃权'}
        return {f'我投{min(seat for seat, score in scores.items() if score == maximum)}号'}
    scores = {seat: sum(attr in pair['civilian_attrs'] for attr in attrs)
              for seat, attrs in used.items() if seat != sample['meta']['my_seat']}
    minimum = min(scores.values())
    # The original generator randomly abstains with probability 0.2.
    return {'我弃权', f'我投{min(seat for seat, score in scores.items() if score == minimum)}号'}


def invalid_reason(sample, text):
    if text == '我弃权':
        return None
    match = re.fullmatch(r'我投([0-9]+)号', text)
    if not match:
        return 'syntax_or_extra_text'
    seat = int(match[1])
    meta = sample['meta']
    if seat == meta['my_seat']:
        return 'self_vote'
    if not 1 <= seat <= meta['n_players']:
        return 'out_of_range'
    if 'alive_seats' in meta and seat not in meta['alive_seats']:
        return 'eliminated_seat'
    return None


def audit(samples, reports):
    counts = Counter()
    details = []
    for index, sample in enumerate(samples):
        if sample['task'] != 'vote':
            continue
        folded = '轮略）' in sample['messages'][0]['content']
        bucket = 'folded' if folded else 'unfolded'
        counts[f'{bucket}_votes'] += 1
        target = sample['messages'][1]['content']
        observable = visible_rule_targets(sample)
        if target not in observable:
            counts[f'{bucket}_label_disagrees_with_visible_rule'] += 1
        row = {'index': index, 'id': sample['id'], 'role': sample['role'],
               'folded': folded, 'target': target,
               'visible_rule_compatible_targets': sorted(observable), 'conditions': {}}
        for name, report in reports.items():
            record = report['records'][index]
            if record['target'] != target:
                raise ValueError('Report target order mismatch')
            output = record['output']
            reason = invalid_reason(sample, output)
            counts[f'{name}_votes'] += 1
            counts[f'{name}_label_match'] += output == target
            counts[f'{name}_{reason or "legal_vote"}'] += 1
            if sample['role'] == '平民':
                counts[f'{name}_civilian_votes'] += 1
                counts[f'{name}_civilian_undercover_hit'] += output == f"我投{sample['meta']['uc_seat']}号"
            row['conditions'][name] = {'output': output, 'invalid_reason': reason}
        if len(reports) == 2:
            outputs = [value['output'] for value in row['conditions'].values()]
            counts['paired_vote_outputs_unchanged'] += outputs[0] == outputs[1]
        details.append(row)
    return {'counts': dict(counts), 'details': details,
            'limitations': ['Visible-rule reconstruction uses the hidden pair attribute dictionary as an oracle.',
                            'A matching rule target is not proof of optimal game strategy.',
                            'Undercover labels include a random abstention draw absent from the input.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.data.open()]
    reports = {name: json.loads(path.read_text()) for name, path in [('control', args.control), ('state', args.state)]}
    if any(len(report['records']) != len(samples) for report in reports.values()):
        raise ValueError('Report lengths differ from data')
    result = audit(samples, reports)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['counts'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
