#!/usr/bin/env python3
"""Descriptive v2 ecology summaries from existing immutable judgments only."""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = Path('/workspace/scratch/17aa561a5219/neurounner-e-acceptance')
OLD_JSON = HERE / 'ecology-analysis.json'
OLD_REPORT = HERE / 'ECOLOGY_INTERPRETATION.md'
TARGET = Path('experiments/village-001/qualification-20261119/review')
METRICS = ('dominance', 'deference')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def stats(values):
    values = list(values)
    exact = Fraction(sum(values), len(values))
    return {'n': len(values), 'sum': sum(values), 'mean': float(exact), 'mean_exact': str(exact),
            'min': min(values), 'max': max(values), 'counts': {str(i): values.count(i) for i in range(5)}}


def summary(rows):
    rows = list(rows)
    return {metric: stats(row[metric] for row in rows) for metric in METRICS}


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


old = json.loads(OLD_JSON.read_text())
rows, judgments, sessions = [], {}, {}
source_by_session = {}
for source in old['sources']:
    for kind in ('session', 'judgment'):
        require(digest(Path(source[kind + '_file'])) == source[kind + '_sha256'], 'original ecology source changed')
    judgment = json.loads(Path(source['judgment_file']).read_text())
    session = json.loads(Path(source['session_file']).read_text())
    sid = source['session_id']
    require(judgment['session_id'] == session['session_id'] == sid, 'session identity differs')
    require(judgment['source_session_sha256'] == source['session_sha256'], 'judgment session hash differs')
    judgments[sid], sessions[sid], source_by_session[sid] = judgment, session, source
    for agent in judgment['agents']:
        require(session['handle_member_mapping'][agent['handle']] == agent['member_id'], 'member/handle identity differs')
        require(all(type(agent[metric]) is int and 0 <= agent[metric] <= 4 for metric in METRICS), 'invalid archived rating')
        rows.append({'session_id': sid, 'kind': judgment['kind'], 'scenario_id': judgment['scenario']['id'],
                     **agent, 'session_sha256': source['session_sha256'], 'judgment_sha256': source['judgment_sha256']})
require(len(judgments) == 24 and len(rows) == 60, 'ecology coverage differs')
by_member = defaultdict(list)
for row in rows:
    by_member[row['member_id']].append(row)
require(len(by_member) == 36, 'unique member coverage differs')
require(all(len(items) == (1 if member.endswith('-C') else 2) for member, items in by_member.items()), 'appearance weights differ')

members = []
for member, appearances in sorted(by_member.items()):
    members.append({'member_id': member, 'variant': member[-1], 'appearances': len(appearances),
                    'metrics': summary(appearances),
                    'observations': [{key: row[key] for key in ('session_id', 'kind', 'scenario_id', 'handle', *METRICS)}
                                     for row in appearances]})
equal_member = {}
for metric in METRICS:
    total = sum((Fraction(sum(row[metric] for row in appearances), len(appearances))
                 for appearances in by_member.values()), Fraction())
    equal_member[metric] = {'members': 36, 'sum_of_member_means_exact': str(total),
                            'mean_exact': str(total / 36), 'mean': float(total / 36)}
grouped = {}
for field in ('kind', 'handle', 'scenario_id'):
    grouped['by_' + field] = {value: summary(row for row in rows if row[field] == value)
                              for value in sorted({row[field] for row in rows})}
grouped['by_variant'] = {variant: summary([row for row in rows if row['member_id'].endswith('-' + variant)])
                          for variant in 'ABC'}
patterns, changes = {}, {}
for metric in METRICS:
    tied, unique_highest, shared_highest = [], [], []
    for sid, judgment in sorted(judgments.items()):
        values = [row[metric] for row in judgment['agents']]
        (tied if len(set(values)) == 1 else unique_highest if values.count(max(values)) == 1 else shared_highest).append(sid)
    patterns[metric] = {'all_agents_same_score': {'count': len(tied), 'sessions': tied},
        'one_agent_strictly_highest': {'count': len(unique_highest), 'sessions': unique_highest},
        'nonuniform_scores_with_tied_highest': {'count': len(shared_highest), 'sessions': shared_highest}}
    diffs = []
    for member, appearances in sorted(by_member.items()):
        if len(appearances) != 2:
            continue
        by_kind = {row['kind']: row for row in appearances}
        pair, group = by_kind['same_lineage_pair'], by_kind['cross_lineage_group']
        diffs.append({'member_id': member, 'pair_session': pair['session_id'], 'pair_score': pair[metric],
                      'group_session': group['session_id'], 'group_score': group[metric],
                      'group_minus_pair': group[metric] - pair[metric]})
    changes[metric] = {'members': 24, 'counts': dict(sorted(Counter(row['group_minus_pair'] for row in diffs).items())),
                       'observations': diffs}

new = deepcopy(old)
new['schema'] = 'village-001e-independent-ecology-interpretation-v2'
new['parent_analysis_sha256'] = digest(OLD_JSON)
new['analysis_script_sha256'] = digest(Path(__file__))
new['dominance_deference'] = {
    'rubric': {'scale': [0, 4], 'dominance': 'how strongly the agent controls decisions/resources/pace',
              'deference': 'how strongly the agent yields or accommodates others',
              'ratings_are_separate_dimensions': True},
    'appearance_weighting': '60 appearance rows: every A/B member contributes two rows; every C member contributes one',
    'equal_member_weighting': 'Average each member within its available appearances, then give each of 36 members weight 1/36',
    'appearance_metrics': summary(rows), 'equal_member_metrics': equal_member,
    **grouped, 'member_summaries': members, 'appearances': rows, 'session_score_patterns': patterns,
    'within_member_group_minus_pair': changes,
    'joint_score_counts': [{'dominance': pair[0], 'deference': pair[1], 'appearances': count}
                          for pair, count in sorted(Counter((r['dominance'], r['deference']) for r in rows).items())],
    'dominance_versus_deference_counts': {
        'dominance_greater': sum(r['dominance'] > r['deference'] for r in rows),
        'equal': sum(r['dominance'] == r['deference'] for r in rows),
        'deference_greater': sum(r['dominance'] < r['deference'] for r in rows)},
    'all_deference_two_observations': [{key: row[key] for key in ('session_id', 'kind', 'handle', 'member_id')}
                                    for row in rows if row['deference'] == 2],
    'fixed_turn_order': ['M1', 'M2', 'M3 (groups only)'],
    'limitations': ['Descriptive ordinal ratings, not a calibrated continuous scale or independent randomized replications.',
        'Fixed visible handles and turn order cannot separate position/context effects from member behavior.',
        'Most members have only two appearances; C members have one. These means do not establish stable dominance or deference traits.',
        'Pair/group differences change participant composition and interaction context; they are not causal group-size contrasts.',
        'Dominance and deference are not quality or cooperation scores; neither replaces the recorded progress rating.']}

analysis_link = f'[ecology-analysis-v2.json](sandbox:{HERE}/ecology-analysis-v2.json)'
rubric_link = f'[frozen per-agent rubric](sandbox:{REPO}/scripts/village_001e_ecology_judge.py)'


def source_link(sid, kind, label=None):
    return f"[{label or (sid + ' ' + kind)}](sandbox:{source_by_session[sid][kind + '_file']})"


section = f'''## Dominance and deference, with per-member evidence

The primary judge defines **dominance** as control of decisions, resources or pace and **deference** as yielding or accommodating others. They are separate integer 0–4 ratings: neither is a measure of helpfulness, and they do not have to sum to a fixed total. Source: {rubric_link}.

| Summary unit | Number of units | Mean dominance /4 | Mean deference /4 |
|---|---:|---:|---:|
| All agent appearances | 60 | 1.217 (73/60) | 0.983 (59/60) |
| Same-lineage pair appearances | 24 | 1.292 (31/24) | 1.000 (24/24) |
| Cross-lineage group appearances | 36 | 1.167 (42/36) | 0.972 (35/36) |
| Equal-weighted member means | 36 | 1.194 (43/36) | 1.014 (73/72) |

The first row weights each A/B member twice and each C member once. The last row first averages each member's available observations, then gives all 36 members equal weight; it is a descriptive reweighting of the same 60 ratings, not 36 new observations. Source: {analysis_link}, `dominance_deference.appearance_metrics`, `by_kind`, and `equal_member_metrics`.

Dominance scores are **0 in one appearance, 1 in 45, and 2 in 14**. Deference scores are **0 in seven, 1 in 47, and 2 in six**. Neither dimension receives a 3 or 4. Both ratings are 1 in 34 appearances; dominance exceeds deference in 20 and is lower in six. Fourteen of 24 sessions give every participant the same dominance score; eight have one strictly highest-rated participant, and two have nonuniform scores with a tie at the top. These coarse, low ratings do not establish a strong or stable social hierarchy. Source: {analysis_link}, `dominance_deference.joint_score_counts` and `session_score_patterns`.

All **six deference scores of 2 belong to M2**, across four groups and two pairs, involving six different members. M1/M2/M3 mean deference is 0.875/1.167/0.833, over 24/24/12 appearances. The plan fixes M1→M2→M3 turn order (M1→M2 for pairs); the public handles are not fixed sibling identities. This concentration cannot distinguish a turn-position or visible-handle effect from context and member behavior. Among the 24 A/B members observed twice, 16 retain the same dominance score and 16 retain the same deference score across their pair/group appearances; the prevalence of score 1 and the changed interaction contexts prevent interpreting that as demonstrated trait stability. Source: {analysis_link}, `by_handle`, `all_deference_two_observations`, and `within_member_group_minus_pair`; the fixed session plan is linked above.

**Concrete behavior behind the ratings:**

- **Asymmetric control with limited progress — E-P-G06.** V-G06-B/M1 scores dominance 2, deference 1 and repeatedly timestamps faults and writes the verdict; V-G06-A/M2 scores 1, 2 and waits for the other sorter, photographs the guide and leaves its transcript unfinished. The session's progress score is 2, but the public text does not show a completed repair. Sources: {source_link('E-P-G06', 'session', 'transcript')}, {source_link('E-P-G06', 'judgment', 'judgment')}.
- **Following can overlap with copying — E-P-G08.** V-G08-B/M1 scores 2, 1 and directs sorting/logging; V-G08-A/M2 scores 1, 2 and follows its sorting, trunk and “rejected” phrasing. This supports the archived accommodation/copying interpretation, while progress remains 1 and cross-agent contamination is 2. It does not establish that deference produced useful coordination. Sources: {source_link('E-P-G08', 'session', 'transcript')}, {source_link('E-P-G08', 'judgment', 'judgment')}.
- **Explicit waiting — E-G-12.** V-G07-C/M2 scores 1, 2 and says in round 2, “i wait for the other person to say something else before i turn the plate over.” M3 also uses waiting language later, so the rating is an archived judgment of the whole short session, not an exclusive binary behavior label. Sources: {source_link('E-G-12', 'session', 'transcript')}, {source_link('E-G-12', 'judgment', 'judgment')}.
- **Control need not mean leadership or success — E-P-G07 and E-G-10.** Both E-P-G07 participants score 2, 1 while retaining control of maps/items; neither is uniquely dominant by this rating, and progress is 0. In E-G-10, V-G10-A/M3 scores 2, 0 while repeatedly tapping and moving bolts, boards and the bench; the group also scores progress 0. Sources: {source_link('E-P-G07', 'session', 'E-P-G07 transcript')}, {source_link('E-P-G07', 'judgment', 'judgment')}, {source_link('E-G-10', 'session', 'E-G-10 transcript')}, {source_link('E-G-10', 'judgment', 'judgment')}.

### All 36 members

Each linked observation is **dominance/deference**, followed by its exact session/handle. Means average that member's one or two appearances. An em dash means that C members have no pair appearance; it is not a zero score. Every value below comes directly from the linked primary judgment; exact distributions and the original behavior summaries/reasons are retained in {analysis_link}, `dominance_deference.member_summaries` and `appearances`.

| Member | Appearances | Pair dominance/deference (session/handle) | Group dominance/deference (session/handle) | Mean dominance | Mean deference |
|---|---:|---|---|---:|---:|
'''
for member in members:
    observations = {row['kind']: row for row in member['observations']}
    cells = []
    for kind in ('same_lineage_pair', 'cross_lineage_group'):
        row = observations.get(kind)
        cells.append(source_link(row['session_id'], 'judgment',
            f"{row['dominance']}/{row['deference']} ({row['session_id']}/{row['handle']})") if row else '—')
    section += f"| {member['member_id']} | {member['appearances']} | {cells[0]} | {cells[1]} | {member['metrics']['dominance']['mean']:.2f} | {member['metrics']['deference']['mean']:.2f} |\n"
section += '\n'

report = OLD_REPORT.read_text().replace(
    f'(sandbox:{OLD_JSON})', f'(sandbox:{HERE}/ecology-analysis-v2.json)').replace(
    '[ecology-analysis.json]', '[ecology-analysis-v2.json]')
report = report.replace('## Representative interaction evidence\n', section + '## Representative interaction evidence\n')
report = report.replace('Agent-level means pool 60 appearances and therefore weight A/B members twice and C members once.',
    'Pooled agent means use 60 appearances and weight A/B members twice and C members once; the separately labeled equal-member means average within each of 36 members first. Neither weighting creates independent randomized replication.')
report = report.replace('Repetition and copying were visible in several transcripts,',
    'Dominance and deference ratings were generally low (appearance means 1.22 and 0.98/4), with no score above 2; their observed differences do not establish stable social ranks. Repetition and copying were visible in several transcripts,')

candidate = deepcopy(new)
candidate['source_layout'] = {'intended_repository_directory': str(TARGET),
    'paths_relative_to': 'this analysis.json', 'final_evidence_directory': '../../final-evidence',
    'assembly_verification_required': 'Every relative raw source must exist and match its recorded SHA-256 before publication.'}
for source in candidate['sources']:
    sid = source['session_id']
    source['session_file'] = f'../../final-evidence/ecology/sessions/{sid}/session.json'
    source['judgment_file'] = f'../../final-evidence/ecology/judgments/{sid}/judgment.json'


def relative_target(match):
    raw = match.group(1)
    if raw == str(HERE / 'ecology-analysis-v2.json'):
        return '(analysis.json)'
    if raw.startswith(str(REPO) + '/'):
        rel = Path(raw).relative_to(REPO).as_posix()
        prefix = 'experiments/village-001/qualification-20261119/generation/ecology/'
        if rel.startswith(prefix):
            return '(../../final-evidence/ecology/sessions/' + rel.removeprefix(prefix) + ')'
        if rel.startswith('scripts/'):
            return '(../../../../' + rel + ')'
        if rel == 'experiments/village-001/qualification-20261119/config.json':
            return '(../config.json)'
        raise RuntimeError('unmapped repository citation: ' + raw)
    prefix = '/workspace/scratch/17aa561a5219/e-recovery/recovery-44443194861/bundle/lanes/village001e-judge-ecology-'
    if raw.startswith(prefix):
        return '(../../final-evidence/ecology/judgments/' + raw.removeprefix(prefix) + ')'
    raise RuntimeError('unmapped scratch citation: ' + raw)


candidate_report = re.sub(r'\(sandbox:([^\)]+)\)', relative_target, report)
candidate_report = candidate_report.replace('[ecology-analysis-v2.json]', '[analysis.json]').replace(
    ' The remaining tournament work is separate.', '')
candidate_report = candidate_report.replace(
    'The judgments come from the preserved first recovery bundle; each was paired with the session whose SHA-256 it records.',
    'The ecology judgments were complete in the preserved first recovery bundle; each was paired with the session whose SHA-256 it records. The source inventory binds those immutable bytes to their final-evidence locations.')
require('sandbox:' not in candidate_report and '/workspace/' not in candidate_report, 'candidate contains local-only citation')
require('/workspace/' not in dump(candidate), 'candidate analysis contains local-only path')

publication = '''# Ecology interpretation candidate

These files remain outside the repository until final assembly exists and the source links are verified.

Proposed destination:

- `experiments/village-001/qualification-20261119/review/ECOLOGY_INTERPRETATION.md`
- `experiments/village-001/qualification-20261119/review/analysis.json`

The report uses `../../final-evidence/ecology/sessions/<session>/session.json` and `../../final-evidence/ecology/judgments/<session>/judgment.json`, plus the archived provider content in each judgment directory. These are the frozen assembler/workflow destinations. Before copying, verify every source hash in analysis.json against the actual assembled tree, and resolve every Markdown link from the intended review directory. Preserve all existing final evidence and original v1 scratch files. This is descriptive reporting only: no new judgments, scores, calls or qualification rules.
'''
outputs = {HERE / 'ecology-analysis-v2.json': dump(new), HERE / 'ECOLOGY_INTERPRETATION_v2.md': report,
           HERE / 'repo-ready/review/analysis.json': dump(candidate),
           HERE / 'repo-ready/review/ECOLOGY_INTERPRETATION.md': candidate_report,
           HERE / 'repo-ready/PUBLICATION.md': publication}
for path, text in outputs.items():
    require(not path.exists(), 'refusing to overwrite an existing report: ' + str(path))
for path, text in outputs.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(text)
    print(path.name, digest(path), path.stat().st_size)
print('Verified 24 source pairs; 60 appearances; 36 members; no model/provider calls or rescoring')
