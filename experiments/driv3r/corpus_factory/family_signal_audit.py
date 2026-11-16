#!/usr/bin/env python3
"""Read-only audit of genuine family-conditioned VISUAL diversity in v1 game corpus.

Three different family tokens do not imply three different drawings.  This
uses the exact PNG-render inputs (painted set, branch set, final tip) rather
than counting names or policy-only hashes as visual diversity.  Does not
promote synthetic supervision to native evidence or create model labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from corpus import SCENARIOS, SPLITS, SOURCE, VERSION, audit, canonical, partition, episode as regenerate_episode

FAMILIES = frozenset(('greebles', 'pipework', 'architecture'))


def render_signature(episode):
    """Exactly the episode-dependent inputs to corpus.png_bytes()."""
    trail = episode['trail']
    if not trail:
        raise ValueError('missing source trail')
    return hashlib.sha256(canonical({
        'painted': sorted({tuple(p) for p in trail}),
        'branches': sorted({tuple(p) for p in episode['branch_points']}),
        'tip': trail[-1],
    }).encode()).hexdigest()


def analyze(root):
    root = Path(root)
    checked = audit(root)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != VERSION or manifest.get('source') != SOURCE:
        raise ValueError('unrecognized pinned synthetic source')
    worlds = {}
    for split in SPLITS:
        for line in (root / f'{split}.worlds.jsonl').read_text().splitlines():
            w = json.loads(line)
            group = w['world_group_id']
            if group in worlds or w.get('split') != split or partition(group) != split:
                raise ValueError('duplicate or mispartitioned world')
            worlds[group] = w
    groups = defaultdict(dict)
    for split in SPLITS:
        for line in (root / f'{split}.episodes.jsonl').read_text().splitlines():
            ep = json.loads(line)
            group, family = ep['world_group_id'], ep['family']
            w = worlds.get(group)
            if (w is None or ep.get('split') != split or ep.get('world_id') != w['world_id']
                    or ep.get('scenario') != w['scenario'] or family not in FAMILIES
                    or family in groups[group] or len(ep.get('trail', [])) != ep['steps']+1):
                raise ValueError('corrupt world/episode family lineage')
            if ep.get('policy') != 'heuristic':
                raise ValueError('non-heuristic source episode')
            replay_w, replay_ep, _ = regenerate_episode(
                w['generator_seed'], w['scenario'], family,
                policy='heuristic', budget=ep['step_budget'])
            if replay_w != w or replay_ep != ep:
                raise ValueError('episode differs from pinned deterministic producer')
            groups[group][family] = ep
    if set(groups) != set(worlds):
        raise ValueError('world missing episodes')
    per_scenario = defaultdict(Counter)
    per_split = defaultdict(Counter)
    total = Counter()
    for group, family_episodes in groups.items():
        if set(family_episodes) != FAMILIES:
            raise ValueError('not exactly three source families')
        w = worlds[group]
        signatures = {render_signature(ep) for ep in family_episodes.values()}
        distinct = len(signatures)
        if not 1 <= distinct <= 3:
            raise ValueError('invalid render diversity')
        for counter in (total, per_scenario[w['scenario']], per_split[w['split']]):
            counter['worlds'] += 1
            counter[f'exact_render_variants_{distinct}'] += 1
            counter['family_episodes'] += 3
            counter['goal_reaching_episodes'] += sum(bool(ep['goal_reached']) for ep in family_episodes.values())
            counter['total_decisions'] += sum(ep['steps'] for ep in family_episodes.values())
    if set(per_scenario) != set(SCENARIOS) or len(worlds) != checked['world_groups']:
        raise ValueError('scenario/world population inconsistency')
    result = {
        'schema': 'cybersand.driv3r.corpus-family-visual-diversity/1',
        'evidence_class': 'synthetic-heuristic-source-episode-replay',
        'native_game_records': 0, 'model_optimizer_steps': 0,
        'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'source_audit': checked,
        'render_equivalence': 'same visited-cell set, branch-cell set, final tip within one source world',
        'total': dict(sorted(total.items())),
        'by_scenario': {k: dict(sorted(v.items())) for k,v in sorted(per_scenario.items())},
        'by_split': {k: dict(sorted(v.items())) for k,v in sorted(per_split.items())},
        'warning': 'Family names are NOT evidence of family-distinct renderings. '
                   'Keep this synthetic teacher separate from native and model evaluation.',
    }
    return result


def main():
    p = argparse.ArgumentParser(description='Source-checked, read-only family visual-diversity audit')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, output = args.corpus.resolve(), args.output.resolve()
    if source == output or source in output.parents or output.exists():
        raise ValueError('immutable output must be new and outside the corpus')
    result = analyze(source)
    output.write_text(canonical(result) + '\n')
    print(canonical(result))

if __name__ == '__main__':
    main()
