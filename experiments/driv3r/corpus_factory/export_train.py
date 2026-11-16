#!/usr/bin/env python3
"""Emit TRAIN-only, prompt-safe WORM action examples from an audited synthetic corpus.

Do not pass source decision JSONL rows directly to a model: they contain labels,
oracle scores and receipts. No native/Workbench evidence or model training here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from corpus import (ACTIONS, SOURCE, VERSION, audit, canonical, digest,
                    partition, world as regenerate_world, episode as regenerate_episode,
                    local_grid)

FIELDS = (
    'x', 'y', 'heading', 'local_5x5', 'legal', 'last_actions', 'family',
    'tick', 'local_material_5x5', 'local_protected_5x5', 'goal_delta',
    'goal_reached', 'budget_remaining',
)
SYSTEM = (
    'You steer a synthetic CyberSand-style 96x96 drawing worm. '
    'Select exactly ONE action from the host-provided legal menu. '
    'Material/protection/goal fields are experimental synthetic channels, '
    'not a native Godot contract. Answer with the action token only.'
)


def observed_input(obs):
    """Positive allowlist: never copy full teacher rows into prompts."""
    if any(k not in obs for k in FIELDS):
        raise ValueError('missing required observable input field')
    data = {key: obs[key] for key in FIELDS}
    if not isinstance(data['legal'], list) or not data['legal']:
        raise ValueError('missing host legal menu')
    if len(data['legal']) != len(set(data['legal'])) or any(a not in ACTIONS for a in data['legal']):
        raise ValueError('invalid host legal menu')
    if len(data['local_5x5']) != 5 or any(len(line) != 5 for line in data['local_5x5']):
        raise ValueError('invalid local grid')
    if len(data['local_material_5x5']) != 5 or any(len(line) != 5 for line in data['local_material_5x5']):
        raise ValueError('invalid local material grid')
    if len(data['local_protected_5x5']) != 5 or any(len(line) != 5 for line in data['local_protected_5x5']):
        raise ValueError('invalid local protection grid')
    return data


def observed_fingerprint(obs):
    """Exactly the v1 producer's pre-action input fingerprint."""
    return digest([
        obs['local_5x5'], obs['local_material_5x5'],
        obs['local_protected_5x5'], obs['goal_delta'],
        obs['heading'], obs['family'], obs['legal'], obs['last_actions'],
    ])


def validate_row(row, train_groups):
    if (row.get('schema') != VERSION or row.get('split') != 'train'
            or row.get('world_group_id') not in train_groups
            or row.get('evidence_class') != 'synthetic-host-heuristic'
            or row.get('target_origin') != 'deterministic-synthetic-heuristic-not-native'):
        raise ValueError('non-TRAIN or non-synthetic source refused')
    obs = observed_input(row['observation'])
    if row.get('input_sha256') != observed_fingerprint(obs):
        raise ValueError('observation fingerprint changed')
    action = row['target_action']
    if action not in obs['legal'] or action not in ACTIONS:
        raise ValueError('non-host-legal supervised label')
    if row.get('receipt', {}).get('applied') != action:
        raise ValueError('inconsistent host action receipt')
    return obs, action



def validate_lineage(row, obs, action, worlds, episodes, rendered_worlds):
    """Cross-check *all* prompt fields against immutable audited world/episode receipts.

    The producer's v1 input_sha256 covers model-local features but omits x/y,
    tick, remaining budget and goal-reached status. Those omitted values must
    not be accepted merely because an attacker rehashed the source manifest.
    """
    group = row['world_group_id']
    world = worlds.get(group)
    ep = episodes.get(row.get('episode_id'))
    if world is None or ep is None:
        raise ValueError('missing world/episode provenance')
    tick = obs['tick']
    if type(tick) is not int or not 0 <= tick < ep['steps']:
        raise ValueError('invalid source episode tick')
    if (row.get('record_id') != f"{ep['episode_id']}:{tick:04d}"
            or row.get('world_id') != world['world_id']
            or ep.get('world_id') != world['world_id']
            or ep.get('world_group_id') != group
            or ep.get('split') != 'train'
            or ep.get('scenario') != world['scenario']
            or row.get('scenario') != world['scenario']
            or row.get('family') != ep.get('family')
            or obs['family'] != ep.get('family')):
        raise ValueError('inconsistent world/episode source lineage')
    if (type(obs['x']) is not int or type(obs['y']) is not int
            or not 0 <= obs['x'] < 96 or not 0 <= obs['y'] < 96
            or type(obs['heading']) is not int or not 0 <= obs['heading'] < 4
            or type(obs['budget_remaining']) is not int
            or obs['budget_remaining'] != ep['step_budget'] - tick):
        raise ValueError('inconsistent observable position/heading/budget')
    if (len(obs['last_actions']) > 4
            or any(a not in ACTIONS for a in obs['last_actions'])
            or obs['local_5x5'][2][2] != '@'
            or 'STOP' not in obs['legal']):
        raise ValueError('invalid observable WORM host state')
    before, after = ep['trail'][tick], ep['trail'][tick + 1]
    receipt = row['receipt']
    if ([obs['x'], obs['y']] != before
            or receipt.get('position_before') != before
            or receipt.get('position_after') != after
            or receipt.get('raw_choice') != action
            or receipt.get('raw_legal') is not True):
        raise ValueError('observation/trail/receipt divergence')
    delta = [world['goal'][0] - obs['x'], world['goal'][1] - obs['y']]
    if (obs['goal_delta'] != delta
            or type(obs['goal_reached']) is not bool
            or obs['goal_reached'] != (before == world['goal'])):
        raise ValueError('inconsistent observed goal information')
    # A manifest/row SHA can be recomputed after accidental or deliberate
    # edits. Verify that every local material/protection/symbolic channel
    # reflects the deterministic original world and painted trail, rather
    # than merely trusting a self-consistent (rehashable) observation.
    obstacles, protected, materials = rendered_worlds[group]
    visited = {tuple(p) for p in ep['trail'][:tick + 1]}
    symbolic, material, immutable = local_grid(
        (obs['x'], obs['y']), visited, obstacles, protected, materials, 96)
    if (obs['local_5x5'] != symbolic
            or obs['local_material_5x5'] != material
            or obs['local_protected_5x5'] != immutable):
        raise ValueError('local observation disagrees with reconstructed world/trail')

def load_train(root):
    root = Path(root)
    audited = audit(root)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != VERSION or manifest.get('source') != SOURCE:
        raise ValueError('unrecognised corpus source/version')
    if manifest.get('claims', {}).get('trained_model_records') != 0:
        raise ValueError('trained-model trace contamination')
    groups = set()
    worlds = {}
    rendered_worlds = {}
    for line in (root / 'train.worlds.jsonl').read_text().splitlines():
        w = json.loads(line)
        group = w['world_group_id']
        if (w.get('split') != 'train'
                or w.get('evidence_class') not in (None, 'synthetic-host-heuristic')
                or partition(group) != 'train'):
            raise ValueError('non-TRAIN world partition')
        if group in groups:
            raise ValueError('duplicate TRAIN world group')
        original, obstacles, protected, materials, _ = regenerate_world(
            w['generator_seed'], w['scenario'])
        if (any(w.get(key) != value for key, value in original.items())
                or w.get('world_id') != digest(original)[:24]
                or w.get('world_group_id') != f"{w['scenario']}:{w['generator_seed']}"):
            raise ValueError('TRAIN world fixture differs from deterministic generator')
        groups.add(group)
        worlds[group] = w
        rendered_worlds[group] = (obstacles, protected, materials)
    episodes = {}
    for line in (root / 'train.episodes.jsonl').read_text().splitlines():
        ep = json.loads(line)
        eid = ep['episode_id']
        if (ep.get('split') != 'train'
                or ep.get('evidence_class') != 'synthetic-host-heuristic'
                or ep.get('world_group_id') not in groups
                or eid in episodes
                or type(ep.get('steps')) is not int
                or type(ep.get('step_budget')) is not int
                or not 0 < ep['steps'] <= ep['step_budget']
                or len(ep.get('trail', [])) != ep['steps'] + 1):
            raise ValueError('invalid TRAIN episode source')
        episodes[eid] = ep
    # Recreate the complete, canonical synthetic teacher trajectories from
    # the producer's pinned seed/scenario/family/budget, not from editable JSONL
    # observations. The v1 source audit is a checksum check, and omitted source
    # fields (heading, legal menu, history, teacher labels) can otherwise be
    # forged together with a newly computed manifest.
    expected_row_hashes = {}
    for ep in episodes.values():
        group = ep['world_group_id']
        w = worlds[group]
        replay_w, replay_ep, replay_rows = regenerate_episode(
            w['generator_seed'], w['scenario'], ep['family'],
            policy='heuristic', budget=ep['step_budget'])
        if replay_w != w or replay_ep != ep:
            raise ValueError('TRAIN episode differs from deterministic producer replay')
        for generated in replay_rows:
            rid = generated['record_id']
            if rid in expected_row_hashes:
                raise ValueError('duplicate producer replay record ID')
            expected_row_hashes[rid] = digest(generated)
    rows = []
    seen = set()
    for line in (root / 'train.decisions.jsonl').read_text().splitlines():
        row = json.loads(line)
        obs, action = validate_row(row, groups)
        validate_lineage(row, obs, action, worlds, episodes, rendered_worlds)
        if expected_row_hashes.get(row.get('record_id')) != digest(row):
            raise ValueError('TRAIN decision differs from deterministic producer replay')
        fingerprint = row['input_sha256']
        if fingerprint in seen:
            raise ValueError('duplicate input in training source')
        seen.add(fingerprint)
        rows.append((row, obs, action))
    replay_stats = {'episodes': len(episodes),
                    'candidate_decisions': len(expected_row_hashes),
                    'matched_train_decisions': len(rows)}
    return rows, hashlib.sha256(manifest_bytes).hexdigest(), audited, replay_stats


def export(root, output, per_action_cap=800, per_world_cap=80, sampler_seed=0):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('export must not overwrite source corpus')
    if per_action_cap < 1 or per_world_cap < 1:
        raise ValueError('selection caps must be positive')
    rows, manifest_sha, audited, replay_stats = load_train(root)
    # Stable worldwide pseudo-random order without replacement. Action and
    # world quotas prevent the STRAIGHT-rich original from monopolising data.
    ranked = sorted(rows, key=lambda triple: hashlib.sha256(
        canonical([manifest_sha, sampler_seed, triple[0]['record_id']]).encode()
    ).hexdigest())
    action_counts = Counter()
    world_counts = Counter()
    selected = []
    for row, obs, action in ranked:
        group = row['world_group_id']
        if action_counts[action] >= per_action_cap or world_counts[group] >= per_world_cap:
            continue
        selected.append((row, obs, action))
        action_counts[action] += 1
        world_counts[group] += 1
    if not selected:
        raise ValueError('no eligible TRAIN examples')
    if output.exists():
        raise FileExistsError('export destination exists; receipts are immutable')
    output.mkdir(parents=True)
    messages = []
    receipts = []
    for index, (row, obs, action) in enumerate(selected):
        messages.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': canonical(obs)},
                {'role': 'assistant', 'content': action},
            ],
        })
        receipts.append({
            'index': index, 'record_id': row['record_id'],
            'world_group_id': row['world_group_id'], 'world_id': row['world_id'],
            'input_sha256': row['input_sha256'], 'label': action,
            'evidence_class': row['evidence_class'],
        })
    inventory = {}
    for filename, records in (
        ('train.messages.jsonl', messages),
        ('train.provenance.jsonl', receipts),
    ):
        raw = ''.join(canonical(record) + '\n' for record in records).encode()
        (output / filename).write_bytes(raw)
        inventory[filename] = {'sha256': hashlib.sha256(raw).hexdigest(),
                               'bytes': len(raw), 'rows': len(records)}
    receipt = {
        'schema': 'cybersand.driv3r.game-corpus-train-export/1',
        'split': 'train-only', 'data_class': 'synthetic-heuristic-supervision',
        'native_game_records': 0, 'model_optimizer_steps': 0,
        'source_manifest_sha256': manifest_sha,
        'source_audit': audited,
        'deterministic_producer_replay': replay_stats,
        'source': SOURCE,
        'selection': {'per_action_cap': per_action_cap,
                      'per_world_cap': per_world_cap, 'sampler_seed': sampler_seed},
        'counts': {'selected': len(selected), 'actions': dict(sorted(action_counts.items())),
                   'world_groups': len(world_counts)},
        'files': inventory,
        'warning': 'Train on train.messages.jsonl only; provenance is analysis-only. '
                   'DEV/heldout and W0 imported evaluation references were never read.',
    }
    (output / 'export-manifest.json').write_text(canonical(receipt) + '\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description='TRAIN-only prompt-safe synthetic game corpus export')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-action-cap', type=int, default=800)
    p.add_argument('--per-world-cap', type=int, default=80)
    p.add_argument('--sampler-seed', type=int, default=0)
    args = p.parse_args()
    print(canonical(export(args.corpus, args.output, args.per_action_cap,
                           args.per_world_cap, args.sampler_seed)))


if __name__ == '__main__':
    main()
