"""Create paired legacy/state-v2 datasets from the original synthetic splits.

These synthetic games have no eliminations, so all initial seats are alive.
Never infer live-game survival from dialogue or apply this conversion to live data.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from prompting import render_player_state


def upgrade_record(record):
    result = copy.deepcopy(record)
    meta = result['meta']
    if 'source' in meta or not result['id'].startswith(('speak-', 'vote-')):
        raise ValueError('Only original synthetic records are supported')
    own, count = meta['my_seat'], meta['n_players']
    if type(count) is not int or type(own) is not int or not 4 <= count <= 6 or not 1 <= own <= count:
        raise ValueError('Invalid synthetic player metadata')
    alive = list(range(1, count + 1))
    extra = render_player_state(own, alive)
    marker = f"你的词语：{result['word']}\n"
    prompt = result['messages'][0]['content']
    if prompt.count(marker) != 1 or '你的座位：' in prompt:
        raise ValueError('Ambiguous or already upgraded prompt')
    result['messages'][0]['content'] = prompt.replace(marker, marker + extra, 1)
    meta.update(alive_seats=alive, prompt_protocol='state-v2')
    label = result['messages'][1]['content']
    if result['task'] == 'vote' and label != '我弃权':
        match = re.fullmatch(r'我投([0-9]+)号', label)
        if not match or int(match[1]) not in set(alive) - {own}:
            raise ValueError('Illegal vote target')
    return result


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control-out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.control_out.exists():
        parser.error('Output directories must not exist; refusing to overwrite')
    if args.out.resolve() == args.control_out.resolve():
        parser.error('State and control outputs must differ')
    outputs = {}
    manifest = {'protocol': 'state-v2', 'alive_source': 'synthetic games without eliminations', 'splits': {}}
    pairs = {}
    prompts = {}
    for split in ['train', 'valid', 'test']:
        blob = (args.source / f'{split}.jsonl').read_bytes()
        old = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
        new = [upgrade_record(row) for row in old]
        pairs[split] = {row['pair_key'] for row in old}
        prompts[split] = {row['messages'][0]['content'] for row in old}
        encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in new).encode()
        outputs[split] = (blob, encoded)
        manifest['splits'][split] = {'rows': len(old), 'source_sha256': digest(blob), 'state_sha256': digest(encoded)}
    if (pairs['train'] | pairs['valid']) & pairs['test']:
        raise ValueError('Held-out word-pair contamination')
    if prompts['train'] & prompts['valid']:
        raise ValueError('Train/validation prompt overlap')
    args.out.mkdir(parents=True)
    args.control_out.mkdir(parents=True)
    for split, (control, state) in outputs.items():
        (args.out / f'{split}.jsonl').write_bytes(state)
        (args.control_out / f'{split}.jsonl').write_bytes(control)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
