#!/usr/bin/env python3
"""Merge synced W&B histories and archive all source experiment artifacts.

Sources are supplied in chronological order; no source files are modified.
Requires `pip install wandb` and `wandb login`.
"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import wandb


def discover_id(folder: Path) -> str:
    candidates = list(folder.glob('wandb/wandb/run-*-*')) + list(folder.glob('wandb/run-*-*'))
    ids = []
    for path in candidates:
        match = re.fullmatch(r'(?:offline-)?run-\d{8}_\d{6}-([a-zA-Z0-9]+)', path.name)
        if match:
            ids.append((path.stat().st_mtime, match.group(1)))
    if not ids:
        raise ValueError(f'No W&B run folder found under {folder}')
    return max(ids)[1]


def scalar(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def archive_experiments(folders, run_ids, output, dry_run=False):
    """Copy checkpoint/config/auxiliary files, preserving all name collisions.

    Identical relative paths are prefixed with the source folder name and run id;
    other files keep their original path. The source manifest maps every file.
    W&B local data is skipped because a fresh W&B run is created in output/wandb.
    """
    entries = []
    for folder, run_id in zip(folders, run_ids):
        for src in sorted(folder.rglob('*')):
            if not src.is_file() or src.is_symlink():
                continue
            rel = src.relative_to(folder)
            if rel.parts[0] == 'wandb':
                continue
            entries.append((folder, run_id, src, rel))

    counts = {}
    for _, _, _, rel in entries:
        counts[rel.as_posix()] = counts.get(rel.as_posix(), 0) + 1

    reserved = set()
    manifest = []
    for folder, run_id, src, rel in entries:
        if counts[rel.as_posix()] > 1:
            # e.g. checkpoints/last__v1-2_q409ta8b.ckpt
            dest_rel = rel.with_name(f'{rel.stem}__{folder.name}_{run_id}{rel.suffix}')
        else:
            dest_rel = rel
        # Covers rare collisions with already-prefixed source filenames.
        candidate = dest_rel
        suffix = 2
        while candidate.as_posix() in reserved:
            candidate = dest_rel.with_name(f'{dest_rel.stem}__{suffix}{dest_rel.suffix}')
            suffix += 1
        dest_rel = candidate
        reserved.add(dest_rel.as_posix())
        manifest.append({'source_experiment': str(folder), 'source_run_id': run_id,
                         'original_path': str(rel), 'merged_path': str(dest_rel),
                         'size_bytes': src.stat().st_size})
        print(f'{"Would copy" if dry_run else "Copy"}: {src} -> {output / dest_rel}')
        if not dry_run:
            dst = output / dest_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if not dry_run:
        (output / 'merge_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(entries)} files {"would be" if dry_run else "were"} archived in {output}')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders', nargs='+', type=Path,
                   help='Experiment directories in chronological order')
    p.add_argument('--entity', required=True, help='W&B username or team')
    p.add_argument('--project', required=True, help='W&B project')
    p.add_argument('--name', default='merged-training', help='New experiment folder and W&B run name')
    p.add_argument('--output', type=Path, help='Output directory (default: sibling of first input named --name)')
    p.add_argument('--dry-run', action='store_true', help='Inspect without uploading or copying files')
    args = p.parse_args()

    folders = [folder.resolve() for folder in args.folders]
    for folder in folders:
        if not folder.is_dir():
            p.error(f'Not a directory: {folder}')
    if args.output:
        output = args.output.resolve()
    else:
        if Path(args.name).name != args.name or args.name in ('.', '..'):
            p.error('--name must be a single directory name, or pass --output')
        output = folders[0].parent / args.name
    if output in folders or any(output.is_relative_to(folder) for folder in folders):
        p.error('Output must not be inside any source experiment directory')
    if output.exists() and any(output.iterdir()):
        p.error(f'Output already exists and is not empty: {output}. Choose --output or --name.')

    api = wandb.Api()
    rows, source_info, run_ids = [], [], []
    for index, folder in enumerate(folders):
        run_id = discover_id(folder)
        run_ids.append(run_id)
        source = api.run(f'{args.entity}/{args.project}/{run_id}')
        count = 0
        for row in source.scan_history():
            if not isinstance(row, dict):
                continue
            clean = {k: v for k, v in row.items()
                     if not k.startswith('_') and scalar(v)}
            if clean:
                clean['source_run_index'] = index
                clean['source_wandb_step'] = row.get('_step') if scalar(row.get('_step')) else count
                rows.append(clean)
                count += 1
        source_info.append({'folder': str(folder), 'id': run_id,
                            'url': source.url, 'points': count})
        print(f'{folder.name}: {run_id} -> {count} scalar history rows')

    if not rows:
        raise RuntimeError('No scalar history found; verify source runs are synced.')
    print(f'Combined: {len(rows)} rows')
    print(json.dumps(source_info, indent=2))
    if args.dry_run:
        archive_experiments(folders, run_ids, output, dry_run=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    archive_experiments(folders, run_ids, output)
    (output / 'merged_sources.json').write_text(json.dumps(source_info, indent=2) + '\n')

    # Local W&B log files are saved beneath the merged experiment directory.
    with wandb.init(entity=args.entity, project=args.project, name=args.name,
                    dir=str(output), job_type='history-merge',
                    config={'merged_from': [info['url'] for info in source_info]},
                    tags=['merged-history']) as target:
        for i, row in enumerate(rows):
            row['merged_step'] = i
            target.log(row, step=i)
        target.summary['source_runs'] = json.dumps(source_info)
        print('New merged run:', target.url)
    print('Merged experiment directory:', output)


if __name__ == '__main__':
    main()
