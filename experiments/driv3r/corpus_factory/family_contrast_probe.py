#!/usr/bin/env python3
"""TRAIN-only family-signal audit and optional ONE-STEP legal contrast probes.

Source: audited synthetic TRAIN chat export, NOT a native game receipt.
Counterfactual family labels are deliberately invented probe targets; they are
never claimed as producer teacher decisions, closed-loop drawings or events.
This script never loads DEV, heldout, W0 evaluation archives or models.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

SCHEMA = 'cybersand.driv3r.family-contrast-probe/1'
SOURCE_SCHEMA = 'cybersand.driv3r.game-corpus-train-export/1'
FAMILIES = ('greebles', 'pipework', 'architecture')
ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'STOP')
INPUT_FIELDS = frozenset(('x', 'y', 'heading', 'local_5x5', 'legal', 'last_actions',
                          'family', 'tick', 'local_material_5x5',
                          'local_protected_5x5', 'goal_delta', 'goal_reached',
                          'budget_remaining'))
REQUIRED = ('export-manifest.json', 'train.messages.jsonl', 'train.provenance.jsonl')
SYSTEM = ('You steer a synthetic CyberSand-style 96x96 drawing worm. '
          'Select exactly ONE action from the host-provided legal menu. '
          'Material/protection/goal fields are experimental synthetic channels, '
          'not a native Godot contract. Answer with the action token only.')


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_export(source):
    """Validate ZIP and original TRAIN prompt/lineage inventory before analysis."""
    raw = Path(source).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(REQUIRED):
            raise ValueError('unexpected or duplicate TRAIN export ZIP members')
        if any(z.getinfo(k).file_size > 32_000_000 for k in REQUIRED):
            raise ValueError('unbounded export ZIP member')
        blobs = {k: z.read(k) for k in REQUIRED}
    manifest = json.loads(blobs['export-manifest.json'])
    if (manifest.get('schema') != SOURCE_SCHEMA or manifest.get('split') != 'train-only'
            or manifest.get('data_class') != 'synthetic-heuristic-supervision'
            or manifest.get('native_game_records') != 0
            or manifest.get('model_optimizer_steps') != 0
            or manifest.get('source_audit', {}).get('audit') != 'PASS'):
        raise ValueError('refused non-TRAIN or unaudited source export')
    for k in REQUIRED[1:]:
        receipt = manifest.get('files', {}).get(k)
        if (not receipt or receipt.get('sha256') != sha(blobs[k])
                or receipt.get('bytes') != len(blobs[k])
                or receipt.get('rows') != len(blobs[k].splitlines())):
            raise ValueError('export file integrity/row count mismatch: ' + k)
    msg_rows = [json.loads(line) for line in blobs[REQUIRED[1]].splitlines()]
    prov_rows = [json.loads(line) for line in blobs[REQUIRED[2]].splitlines()]
    if len(msg_rows) != len(prov_rows) or len(msg_rows) != manifest['counts']['selected']:
        raise ValueError('message/provenance count mismatch')
    seen_ids = set(); observations = []
    for i, (record, provenance) in enumerate(zip(msg_rows, prov_rows)):
        messages = record.get('messages')
        if (not isinstance(messages, list) or len(messages) != 3
                or [m.get('role') for m in messages] != ['system', 'user', 'assistant']
                or messages[0].get('content') != SYSTEM):
            raise ValueError('unexpected source chat-message structure')
        obs = json.loads(messages[1]['content'])
        label = messages[2]['content']
        if (not isinstance(obs, dict) or set(obs) != INPUT_FIELDS
                or obs.get('family') not in FAMILIES or not isinstance(obs['legal'], list)
                or not obs['legal'] or len(set(obs['legal'])) != len(obs['legal'])
                or any(a not in ACTIONS for a in obs['legal'])
                or label not in obs['legal']):
            raise ValueError('invalid or unsafe model-visible observation / choice')
        if (provenance.get('index') != i or provenance.get('label') != label
                or provenance.get('evidence_class') != 'synthetic-host-heuristic'
                or not isinstance(provenance.get('world_group_id'), str)
                or not isinstance(provenance.get('record_id'), str)
                or provenance['record_id'] in seen_ids
                or len(provenance.get('input_sha256', '')) != 64):
            raise ValueError('invalid source TRAIN provenance')
        expected_input_sha = sha(canonical([
            obs['local_5x5'], obs['local_material_5x5'],
            obs['local_protected_5x5'], obs['goal_delta'],
            obs['heading'], obs['family'], obs['legal'], obs['last_actions'],
        ]).encode())
        if provenance['input_sha256'] != expected_input_sha:
            raise ValueError('TRAIN input fingerprint mismatch')
        seen_ids.add(provenance['record_id'])
        observations.append((obs, label, provenance))
    if len({p['world_group_id'] for _, _, p in observations}) != manifest['counts']['world_groups']:
        raise ValueError('TRAIN world group count mismatch')
    return raw, manifest, observations


def family_score(world_observations):
    """Compare same-world, exactly matching pre-action states except family."""
    matched = defaultdict(list); family_counts = Counter(); action_counts = Counter()
    for obs, label, provenance in world_observations:
        stripped = {k: v for k, v in obs.items() if k != 'family'}
        key = (provenance['world_group_id'], canonical(stripped))
        matched[key].append((obs['family'], label, provenance['record_id']))
        family_counts[obs['family']] += 1
        action_counts[label] += 1
    cross = [v for v in matched.values() if len({f for f, _, _ in v}) >= 2]
    divergent = [v for v in cross if len({a for _, a, _ in v}) >= 2]
    return {
        'source_examples': len(world_observations),
        'family_counts': dict(sorted(family_counts.items())),
        'action_counts': dict(sorted(action_counts.items())),
        'matched_same_world_state_groups': len(cross),
        'family_sensitive_action_groups': len(divergent),
        'identical_action_groups': len(cross) - len(divergent),
        'observed_sensitive_fraction': (len(divergent) / len(cross)) if cross else None,
        'divergent_examples': [
            [{'family': f, 'action': a, 'record_id': r} for f, a, r in sorted(v)]
            for v in sorted(divergent, key=canonical)[:12]
        ],
    }


def probe_targets(obs):
    """Transparent, non-executed style probes; never a native/teacher receipt."""
    legal = set(obs['legal'])
    # Restrict to states with unambiguous, genuinely host-legal contrasts.
    if not {'STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH'}.issubset(legal):
        return None
    # Draw many high-frequency junctions, long runs and alternating scaffolds.
    return {'greebles': 'BRANCH', 'pipework': 'STRAIGHT',
            'architecture': 'LEFT' if (obs['tick']//8) % 2 == 0 else 'RIGHT'}


def make_probes(observations, cap=480, per_world=6, selection_seed=0):
    """Optional counterfactual TRAIN-only triples, not observed trajectories.

    Samples are source TRAIN contexts, not additional independently sampled worlds.
    Never use these for scoring model generalization on the same source worlds.
    """
    if not 1 <= cap <= 4096 or not 1 <= per_world <= 64:
        raise ValueError('unbounded probe request')
    candidates = []
    seen = set()
    for obs, original_label, source in observations:
        targets = probe_targets(obs)
        if not targets:
            continue
        base_obs = {k: v for k, v in obs.items() if k != 'family'}
        key = (source['world_group_id'], canonical(base_obs))
        if key in seen:
            continue
        seen.add(key)
        ranking = sha(canonical([selection_seed, key]).encode())
        candidates.append((ranking, base_obs, original_label, obs['family'], source, targets))
    selected = []; per_group = Counter()
    for _, obs, source_label, source_family, source, targets in sorted(candidates, key=lambda x: x[0]):
        group = source['world_group_id']
        if len(selected) == cap:
            break
        if per_group[group] == per_world:
            continue
        per_group[group] += 1
        selected.append((obs, source_label, source_family, source, targets))
    messages = []; provenance = []
    for case_id, (base, original, source_family, source, targets) in enumerate(selected):
        for family in FAMILIES:
            obs = dict(base, family=family)
            action = targets[family]
            if action not in obs['legal']:
                raise ValueError('counterfactual action is not host-legal')
            messages.append({'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': canonical(obs)},
                {'role': 'assistant', 'content': action}]})
            provenance.append({
                'index': len(provenance), 'probe_case': case_id,
                'source_world_group_id': source['world_group_id'],
                'source_record_id': source['record_id'],
                'source_family': source_family,
                'source_label': original, 'counterfactual_family': family,
                'counterfactual_label': action,
                'evidence_class': 'train-synthetic-unexecuted-single-step-style-probe',
                'was_host_applied': False, 'has_native_receipt': False,
            })
    return selected, messages, provenance


def emit(source_zip, output, cap=480, per_world=6, selection_seed=0, probes=False):
    source_raw, src_manifest, observations = load_export(source_zip)
    report = {
        'schema': SCHEMA, 'source_export_zip_sha256': sha(source_raw),
        'source_manifest_sha256': sha(canonical(src_manifest).encode()),
        'source_corpus_manifest_sha256': src_manifest['source_manifest_sha256'],
        'split': 'TRAIN-only source', 'native_game_records': 0,
        'model_optimizer_steps': 0,
        'source_message_audit': 'PASS', 'conditioned_signal': family_score(observations),
        'warning': 'Observed teacher labels are mostly family-insensitive; opt-in '
                   'counterfactual style probes are single-step only, not trajectories '
                   'or evidence of a learned ability.',
    }
    out = Path(output)
    if out.exists():
        raise FileExistsError('immutable output directory already exists')
    if Path(source_zip).resolve() in out.resolve().parents:
        raise ValueError('output inside source ZIP path')
    selected, messages, provenance = make_probes(observations, cap, per_world, selection_seed) if probes else ([], [], [])
    output_bytes = {}
    if probes:
        for name, rows in [('train.family_probes.messages.jsonl', messages),
                           ('train.family_probes.provenance.jsonl', provenance)]:
            output_bytes[name] = (''.join(canonical(row)+'\n' for row in rows)).encode()
    report['probe_selection'] = {
        'enabled': probes, 'seed': selection_seed, 'per_world_cap': per_world,
        'probe_state_cap': cap, 'selected_base_states': len(selected),
        'counterfactual_examples': len(messages),
        'source_group_count': len({p['world_group_id'] for _, _, _, p, _ in selected}),
        'opt_in': 'Non-observed TRAIN preference probe, NEVER evaluate on these contexts',
    }
    report['files'] = {k: {'sha256': sha(b), 'bytes': len(b), 'rows': len(b.splitlines())}
                       for k, b in output_bytes.items()}
    out.mkdir(parents=True)
    for name, payload in output_bytes.items():
        (out/name).write_bytes(payload)
    (out/'family-contrast-manifest.json').write_text(canonical(report)+'\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description='Audit conditional action signal of an immutable TRAIN-only export')
    p.add_argument('source_zip', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--emit-probes', action='store_true')
    p.add_argument('--cap', type=int, default=480)
    p.add_argument('--per-world', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    print(canonical(emit(a.source_zip, a.output, a.cap, a.per_world, a.seed, a.emit_probes)))


if __name__ == '__main__':
    main()
