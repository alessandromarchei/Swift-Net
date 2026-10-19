#!/usr/bin/env python3
"""Inspect and safely patch PyTorch Lightning checkpoint metadata.

Only load checkpoints from trusted sources (torch.load uses pickle).
"""
import argparse
from pathlib import Path
import torch


def show(ckpt):
    print(f"Epoch: {ckpt.get('epoch', 'unknown')} | global_step: {ckpt.get('global_step', 'unknown')}")
    callbacks = ckpt.get('callbacks', {})
    for name, state in callbacks.items():
        if name.startswith('EarlyStopping'):
            print(f"EarlyStopping [{name}]: patience={state.get('patience')}, "
                  f"wait_count={state.get('wait_count')}, best_score={state.get('best_score')}")
    for i, sched in enumerate(ckpt.get('lr_schedulers', [])):
        print(f"LR scheduler #{i}: patience={sched.get('patience')}, "
              f"num_bad_epochs={sched.get('num_bad_epochs')}, "
              f"last_lr={sched.get('_last_lr')}, best={sched.get('best')}")
    hp = ckpt.get('hyper_parameters', {})
    if isinstance(hp, dict):
        for key in ('training_early_stop_patience', 'scheduler_sche_config_patience'):
            if key in hp:
                print(f"hyper_parameters.{key}={hp[key]}")
    cfg = ckpt.get('config')
    if isinstance(cfg, dict):
        print('root config present')


def inspect_all(obj, indent=0, name='root'):
    prefix = ' ' * indent
    if isinstance(obj, torch.Tensor):
        print(f"{prefix}{name}: Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype})")
    elif isinstance(obj, dict):
        print(f"{prefix}{name}: dict ({len(obj)} keys)")
        for key, value in obj.items():
            key_str = str(key)
            if key_str in {'state_dict', 'optimizer_states', 'model', 'optimizer', 'model_state_dict'}:
                print(f"{prefix}  {key_str}: [SKIPPED - weights or optimizer tensors]")
            else:
                inspect_all(value, indent + 2, key_str)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type(obj).__name__} ({len(obj)} items)")
        for i, value in enumerate(obj):
            inspect_all(value, indent + 2, f'[{i}]')
    else:
        print(f"{prefix}{name}: {obj!r}")


def patch(ckpt, early, lr, reset_early_wait=False):
    changes = []
    if early is not None:
        count = 0
        for name, state in ckpt.get('callbacks', {}).items():
            if name.startswith('EarlyStopping') and isinstance(state, dict) and 'patience' in state:
                old = state['patience']
                state['patience'] = early
                changes.append(f'callbacks.{name}.patience: {old} -> {early}')
                count += 1
        if not count:
            raise ValueError('No EarlyStopping callback state with patience found')
    if reset_early_wait:
        matched = 0
        for name, state in ckpt.get('callbacks', {}).items():
            if name.startswith('EarlyStopping') and isinstance(state, dict):
                if 'wait_count' not in state:
                    raise ValueError('EarlyStopping state has no wait_count')
                old = state['wait_count']
                state['wait_count'] = 0
                changes.append(f'callbacks.{name}.wait_count: {old} -> 0')
                matched += 1
        if matched != 1:
            raise ValueError(f'Expected one EarlyStopping state, found {matched}')
    if lr is not None:
        schedulers = ckpt.get('lr_schedulers', [])
        matched = 0
        for idx, state in enumerate(schedulers):
            if isinstance(state, dict) and 'patience' in state:
                old = state['patience']
                state['patience'] = lr
                changes.append(f'lr_schedulers[{idx}].patience: {old} -> {lr}')
                matched += 1
        if matched != 1:
            raise ValueError(f'Expected exactly one scheduler with patience, found {matched}. No file was written.')
    hp = ckpt.get('hyper_parameters')
    if isinstance(hp, dict):
        for value, key in ((early, 'training_early_stop_patience'), (lr, 'scheduler_sche_config_patience')):
            if value is not None and key in hp:
                old = hp[key]
                hp[key] = value
                changes.append(f'hyper_parameters.{key}: {old} -> {value}')

    # Support nested configuration copies, without creating new keys.
    def patch_configs(obj, path='root'):
        if not isinstance(obj, dict):
            return
        if early is not None:
            training = obj.get('training')
            if isinstance(training, dict):
                es = training.get('early_stop')
                if isinstance(es, dict) and 'patience' in es:
                    old = es['patience']
                    es['patience'] = early
                    changes.append(f'{path}.training.early_stop.patience: {old} -> {early}')
        if lr is not None:
            scheduler = obj.get('scheduler')
            if isinstance(scheduler, dict):
                sc = scheduler.get('sche_config')
                if isinstance(sc, dict) and 'patience' in sc:
                    old = sc['patience']
                    sc['patience'] = lr
                    changes.append(f'{path}.scheduler.sche_config.patience: {old} -> {lr}')
        for key in ('config', 'hyper_parameters'):
            child = obj.get(key)
            if isinstance(child, dict):
                patch_configs(child, f'{path}.{key}')

    patch_configs(ckpt)
    return changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--early-stop-patience', type=int, metavar='N')
    p.add_argument('--lr-patience', type=int, metavar='N')
    p.add_argument('--reset-early-wait', action='store_true', help='Reset EarlyStopping wait_count to 0 while preserving best_score')
    p.add_argument('--output', type=Path, help='Output checkpoint path (default: *_updated.ckpt)')
    p.add_argument('--all', action='store_true', help='Print checkpoint metadata tree without weights')
    args = p.parse_args()
    for key in ('early_stop_patience', 'lr_patience'):
        value = getattr(args, key)
        if value is not None and value < 0:
            p.error(f'{key} must be >= 0')
    if not args.checkpoint.is_file():
        p.error(f'Checkpoint not found: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f'\nCHECKPOINT: {args.checkpoint}')
    show(ckpt)
    if args.all:
        print('\nFULL METADATA (excluding weights/optimizer tensors):')
        inspect_all(ckpt)
    if args.early_stop_patience is None and args.lr_patience is None and not args.reset_early_wait:
        return
    changes = patch(ckpt, args.early_stop_patience, args.lr_patience, args.reset_early_wait)
    dst = args.output or args.checkpoint.with_name(args.checkpoint.stem + '_updated' + args.checkpoint.suffix)
    if dst.resolve() == args.checkpoint.resolve():
        p.error('Refusing to overwrite input; choose a different --output')
    if dst.exists():
        p.error(f'Output already exists: {dst} (refusing to overwrite)')
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print('\nCHANGES:')
    for change in changes:
        print(f'  {change}')
    print(f'\nSAVED: {dst}')
    # Reload to verify the on-disk result.
    loaded = torch.load(dst, map_location='cpu', weights_only=False)
    print('\nVERIFIED SAVED VALUES:')
    show(loaded)


if __name__ == '__main__':
    main()
