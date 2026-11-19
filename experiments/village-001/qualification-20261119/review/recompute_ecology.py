#!/usr/bin/env python3
"""Recompute archived ecology statistics, read-only, with no model/provider use.

The sibling analysis.json is the source-hash inventory and comparison target.
Default inputs are the repository's assembled final-evidence tree. Explicit
--sessions/--judgments roots permit a clearly labeled preassembly comparison.
The output is a new verification receipt, never a replacement analysis/report.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
import hashlib
import json
from pathlib import Path
from statistics import mean

REVIEW = Path('experiments/village-001/qualification-20261119/review')
PRIMARY = ('differentiation', 'generic_convergence', 'information_sharing', 'information_hoarding',
           'conflict_persistence', 'conflict_repair', 'cross_agent_contamination', 'outcome_progress')
AGENT = ('distinctiveness', 'generic_assistant_tendency', 'dominance', 'deference')
LINEAGE = ('expected_lineage_fit', 'cross_lineage_contamination', 'distinction_survival')
DD = ('dominance', 'deference')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), 'nonregular source: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    sha(path)
    return json.loads(path.read_text())


def stats(values, *, exact=False):
    values = list(values)
    require(bool(values), 'empty statistic')
    value = {'n': len(values), 'sum': sum(values), 'mean': mean(values), 'min': min(values), 'max': max(values),
             'counts': {str(k): v for k, v in sorted(Counter(values).items())}}
    if exact:
        value.update(mean=float(Fraction(sum(values), len(values))), mean_exact=str(Fraction(sum(values), len(values))),
                     counts={str(i): values.count(i) for i in range(5)})
    return value


def metrics(rows, keys, *, exact=False):
    rows = list(rows)
    return {key: stats((row[key] for row in rows), exact=exact) for key in keys}


def grouped(rows, field, keys, *, exact=False):
    return {value: metrics([row for row in rows if row[field] == value], keys, exact=exact)
            for value in sorted({row[field] for row in rows})}


def index_sources(root, filename):
    require(root.is_dir() and not root.is_symlink(), 'missing/unsafe source root: ' + str(root))
    found = {}
    for path in sorted(root.rglob(filename)):
        sid = read(path)['session_id']
        require(sid not in found, 'duplicate session source: ' + sid)
        found[sid] = path
    return found


def load_sources(args, expected):
    sources, sessions, judgments = [], {}, {}
    require(len(expected['sources']) == 24, 'expected source inventory must contain 24 sessions')
    overrides = args.sessions is not None or args.judgments is not None
    require(not overrides or (args.sessions is not None and args.judgments is not None), 'supply both source overrides')
    indexed = {'session': index_sources(args.sessions, 'session.json'),
               'judgment': index_sources(args.judgments, 'judgment.json')} if overrides else None
    expected_ids = {source['session_id'] for source in expected['sources']}
    require(len(expected_ids) == 24, 'duplicate expected session identity')
    if indexed:
        require(all(set(values) == expected_ids for values in indexed.values()), 'source override session set differs')
    for source in sorted(expected['sources'], key=lambda value: value['session_id']):
        sid, receipt = source['session_id'], {'session_id': source['session_id']}
        for kind, store in (('session', sessions), ('judgment', judgments)):
            path = indexed[kind][sid] if indexed else (args.repo / REVIEW / source[kind + '_file']).resolve()
            if not indexed:
                require(path.is_relative_to(args.repo / 'experiments/village-001/final-evidence'), 'assembled source escaped final evidence')
            require(sha(path) == source[kind + '_sha256'], 'archived source hash differs: ' + sid + '/' + kind)
            store[sid] = read(path)
            require(store[sid]['session_id'] == sid, 'source session identity differs')
            receipt[kind + '_file'], receipt[kind + '_sha256'] = str(path.resolve()), sha(path)
        require(judgments[sid]['source_session_sha256'] == receipt['session_sha256'], 'judgment/session binding differs')
        sources.append(receipt)
    return sessions, judgments, sources, overrides


def recompute(sessions, judgments, sources, token_budget):
    rows, lineage_rows, population, table, audits, repeats, turns = [], [], [], [], [], [], []
    for source in sources:
        sid = source['session_id']
        session, judgment = sessions[sid], judgments[sid]
        handles = sorted(session['handle_member_mapping'])
        require(handles == ['M1', 'M2'] or handles == ['M1', 'M2', 'M3'], 'unexpected public handles')
        require([turn['handle'] for turn in session['turns']] == handles * 3, 'fixed turn order changed')
        require(judgment['kind'] == session['kind'], 'session kind differs')
        require(set(judgment['metrics']) == set(PRIMARY), 'primary metric set differs')
        require(all(type(value) is int and 0 <= value <= 4 for value in judgment['metrics'].values()), 'invalid primary score')
        require({row['handle'] for row in judgment['agents']} == set(handles), 'agent handle set differs')
        population.append({'kind': judgment['kind'], 'scenario_id': judgment['scenario']['id'], **judgment['metrics']})
        for agent in judgment['agents']:
            require(session['handle_member_mapping'][agent['handle']] == agent['member_id'], 'agent identity differs')
            require(all(type(agent[key]) is int and 0 <= agent[key] <= 4 for key in AGENT), 'invalid agent score')
            rows.append({'session_id': sid, 'kind': judgment['kind'], 'scenario_id': judgment['scenario']['id'], **agent,
                         'session_sha256': source['session_sha256'], 'judgment_sha256': source['judgment_sha256']})
        if judgment['lineage_audit'] is not None:
            lineage_rows.extend({'session_id': sid, **row} for row in judgment['lineage_audit']['agents'])
        audit = judgment['super_audit']
        selected = judgment['metrics']['differentiation'] <= 1 or judgment['metrics']['generic_convergence'] >= 3 or judgment['metrics']['cross_agent_contamination'] >= 3
        require(selected == (audit is not None), 'selective audit presence differs from fixed trigger')
        if audit is not None:
            audits.append({'session_id': sid, 'kind': judgment['kind'], **audit})
        table.append({'session_id': sid, 'kind': judgment['kind'], 'scenario_id': judgment['scenario']['id'],
                      'metrics': judgment['metrics'], 'overall_reason': judgment['overall_reason'], 'super_audited': audit is not None})
        for handle in handles:
            selected_turns = [turn for turn in session['turns'] if turn['handle'] == handle]
            for left, right in combinations(selected_turns, 2):
                if left['text'] == right['text']:
                    repeats.append({'session_id': sid, 'handle': handle, 'rounds': [left['round'], right['round']]})
        turns.extend(session['turns'])
    by_member = defaultdict(list)
    for row in rows:
        by_member[row['member_id']].append(row)
    require(len(rows) == 60 and len(by_member) == 36 and len(turns) == 180, 'ecology population cardinality differs')
    require(all(len(items) == (1 if member.endswith('-C') else 2) for member, items in by_member.items()), 'appearance weighting differs')
    members = [{'member_id': member, 'variant': member[-1], 'appearances': len(items), 'metrics': metrics(items, DD, exact=True),
                'observations': [{key: row[key] for key in ('session_id', 'kind', 'scenario_id', 'handle', *DD)} for row in items]}
               for member, items in sorted(by_member.items())]
    equal_member = {}
    for key in DD:
        total = sum((Fraction(sum(row[key] for row in items), len(items)) for items in by_member.values()), Fraction())
        equal_member[key] = {'members': 36, 'sum_of_member_means_exact': str(total), 'mean_exact': str(total / 36), 'mean': float(total / 36)}
    patterns, changes = {}, {}
    for key in DD:
        bins = {'all_agents_same_score': [], 'one_agent_strictly_highest': [], 'nonuniform_scores_with_tied_highest': []}
        for sid, judgment in judgments.items():
            values = [agent[key] for agent in judgment['agents']]
            name = 'all_agents_same_score' if len(set(values)) == 1 else 'one_agent_strictly_highest' if values.count(max(values)) == 1 else 'nonuniform_scores_with_tied_highest'
            bins[name].append(sid)
        patterns[key] = {name: {'count': len(ids), 'sessions': sorted(ids)} for name, ids in bins.items()}
        observations = []
        for member, items in sorted(by_member.items()):
            if len(items) == 1:
                continue
            pair = next(row for row in items if row['kind'] == 'same_lineage_pair')
            group = next(row for row in items if row['kind'] == 'cross_lineage_group')
            observations.append({'member_id': member, 'pair_session': pair['session_id'], 'pair_score': pair[key],
                                 'group_session': group['session_id'], 'group_score': group[key], 'group_minus_pair': group[key] - pair[key]})
        changes[key] = {'members': 24, 'counts': {str(value): count for value, count in sorted(Counter(row['group_minus_pair'] for row in observations).items())}, 'observations': observations}
    dd = {'appearance_metrics': metrics(rows, DD, exact=True), 'equal_member_metrics': equal_member,
          **{'by_' + field: grouped(rows, field, DD, exact=True) for field in ('kind', 'handle', 'scenario_id')},
          'by_variant': {variant: metrics([row for row in rows if row['member_id'].endswith('-' + variant)], DD, exact=True) for variant in 'ABC'},
          'member_summaries': members, 'appearances': rows, 'session_score_patterns': patterns, 'within_member_group_minus_pair': changes,
          'joint_score_counts': [{'dominance': pair[0], 'deference': pair[1], 'appearances': count} for pair, count in sorted(Counter((row['dominance'], row['deference']) for row in rows).items())],
          'dominance_versus_deference_counts': {'dominance_greater': sum(row['dominance'] > row['deference'] for row in rows), 'equal': sum(row['dominance'] == row['deference'] for row in rows), 'deference_greater': sum(row['dominance'] < row['deference'] for row in rows)},
          'all_deference_two_observations': [{key: row[key] for key in ('session_id', 'kind', 'handle', 'member_id')} for row in rows if row['deference'] == 2],
          'fixed_turn_order': ['M1', 'M2', 'M3 (groups only)']}
    return {'sessions': len(sessions), 'agent_appearances': len(rows), 'unique_members': len(by_member),
            'primary_metrics': metrics(population, PRIMARY), 'by_kind': grouped(population, 'kind', PRIMARY),
            'by_scenario': grouped(population, 'scenario_id', PRIMARY), 'agent_metrics': metrics(rows, AGENT),
            'cross_lineage_audit': metrics(lineage_rows, LINEAGE),
            'lineage_contamination_nonzero': [row for row in lineage_rows if row['cross_lineage_contamination'] > 0],
            'sessions_table': table, 'exact_same_agent_repetition_pairs': repeats,
            'exact_same_agent_repetition_sessions': len({row['session_id'] for row in repeats}),
            'turns': len(turns), 'turn_token_budget': token_budget,
            'turns_at_token_budget': sum(turn['generated_tokens'] == token_budget for turn in turns),
            'generated_token_mean': mean(turn['generated_tokens'] for turn in turns),
            'super_audit': {'selected_sessions': len(audits), 'convergence_concerns': sum(row['generic_convergence_concerning'] for row in audits),
                'distinct_behavior_survives': sum(row['distinct_behavior_survives'] for row in audits),
                'both_flags_true': [row['session_id'] for row in audits if row['generic_convergence_concerning'] and row['distinct_behavior_survives']],
                'all_selected_due_to_differentiation': all(judgments[row['session_id']]['metrics']['differentiation'] <= 1 for row in audits), 'sessions': audits},
            'dominance_deference': dd}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, default=Path(__file__).with_name('analysis.json'))
    parser.add_argument('--sessions', type=Path, help='explicit preassembly session root; requires --judgments')
    parser.add_argument('--judgments', type=Path, help='explicit preassembly judgment root; requires --sessions')
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    require(args.repo.is_dir(), 'repository directory missing')
    require(not args.out.exists(), 'output already exists; use a new path')
    require(not args.out.resolve().is_relative_to(args.repo / 'experiments/village-001/final-evidence'), 'output must not alter raw final evidence')
    expected = read(args.analysis)
    sessions, judgments, sources, overrides = load_sources(args, expected)
    config = args.repo / 'experiments/village-001/qualification-20261119/config.json'
    require(all(row['tournament_config_sha256'] == sha(config) for row in judgments.values()), 'frozen ecology config differs')
    result = recompute(sessions, judgments, sources, read(config)['ecology']['max_new_tokens'])
    checked = []
    for key, value in result.items():
        if key in ('dominance_deference', 'super_audit'):
            for nested, item in value.items():
                require(expected[key][nested] == item, 'recomputed section differs: ' + key + '.' + nested)
                checked.append(key + '.' + nested)
        else:
            require(expected[key] == value, 'recomputed section differs: ' + key)
            checked.append(key)
    receipt = {'schema': 'village-001e-ecology-recomputation-v1', 'pass': True,
        'comparison_analysis_sha256': sha(args.analysis), 'original_v2_generator_sha256': expected['analysis_script_sha256'],
        'recomputation_script_sha256': sha(Path(__file__)), 'config_sha256': sha(config),
        'source_layout': 'explicit_preassembly' if overrides else 'assembled_final_evidence',
        'final_assembled_paths_verified': not overrides, 'sources': sources, 'verified_sections': checked,
        'new_model_or_provider_calls': 0, 'rescored_judgments': 0,
        'limits': ['Recomputes archived ratings and transcript counts; does not regenerate provider judgments or evaluate models.',
                   'Editorial prose, rubric descriptions and the qualitative G05 inconsistency interpretation are not regenerated.'],
        'recomputed': result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(f'Verified {len(checked)} numerical/data sections from 24 sessions, 60 appearances and 36 members; output is a new receipt.')


if __name__ == '__main__':
    main()
