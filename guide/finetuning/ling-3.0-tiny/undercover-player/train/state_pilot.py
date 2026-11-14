"""Paired 60-step continuation experiment; validation only, no auto-adoption."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--data-prefix', default='data-v5')
    parser.add_argument('--experiment', default='state-v2-training-pilot')
    parser.add_argument('--adapter-prefix', default='adapters-state-pilot')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    lock = (ROOT / 'eval/.audit-queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = ROOT / 'eval' / args.experiment
    if output.exists():
        raise FileExistsError(output)
    adapters = {name: ROOT / f'{args.adapter_prefix}-{name}' for name in ['control', 'state']}
    for path in adapters.values():
        if path.exists():
            raise FileExistsError(path)
    resume = ROOT / 'adapters-v2-run3/adapters.safetensors'
    fingerprint = sha256(resume)
    sources = {str(path.relative_to(ROOT)): sha256(path)
               for name in adapters for path in (ROOT / f'{args.data_prefix}-{name}').glob('*.jsonl')}
    output.mkdir()
    state = {'status': 'running', 'completed': [], 'steps_per_condition': args.steps,
             'data_prefix': args.data_prefix,
             'resume_sha256': fingerprint, 'data_sha256': sources,
             'limitations': ['Matched optimizer steps, not identical token counts or wall-clock compute.',
                             'Synthetic games have no eliminations; survivor-state coverage is limited.',
                             'Small continuation pilot; not a definitive quality comparison.']}

    def save():
        (output / 'status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')

    def run(label, command):
        if sha256(resume) != fingerprint or any(sha256(ROOT / path) != digest for path, digest in sources.items()):
            raise RuntimeError('Source adapter or data changed during paired experiment')
        state['current'] = label
        save()
        with (output / f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['completed'].append(label)
        save()

    model = str(Path.home() / 'models/Ling-3.0-tiny-4bit')
    save()
    try:
        for name, adapter in adapters.items():
            command = [sys.executable, '-m', 'mlx_lm', 'lora', '--model', model,
                       '--train', '--data', str(ROOT / f'{args.data_prefix}-{name}'),
                       '--fine-tune-type', 'lora', '--optimizer', 'adamw', '--num-layers', '16',
                       '--batch-size', '1', '--grad-accumulation-steps', '4',
                       '--iters', str(args.steps), '--learning-rate', '2e-5',
                       '--max-seq-length', '1024', '--mask-prompt', '--grad-checkpoint',
                       '--adapter-path', str(adapter), '--resume-adapter-file', str(resume),
                       '--save-every', str(args.steps), '--steps-per-report', '10',
                       '--steps-per-eval', '30', '--val-batches', '8', '--seed', '42',
                       '--trust-remote-code']
            run(f'train-{name}', command)
        for name, adapter in adapters.items():
            command = [sys.executable, str(ROOT / 'eval/evaluate.py'), '--model', model,
                       '--adapter-path', str(adapter), '--data', str(ROOT / f'{args.data_prefix}-{name}/valid.jsonl'),
                       '--seed', '42', '--sample-seeds', '--max-tokens', '256',
                       '--out', str(output / f'valid-{name}.json')]
            run(f'valid-{name}', command)
        state.update(status='complete', current=None)
    except Exception as error:
        state.update(status='failed', error=str(error))
        save()
        raise
    save()


if __name__ == '__main__':
    main()
