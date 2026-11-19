#!/usr/bin/env python3
"""Explicit, source-bound E budget recovery without editing frozen E evaluators.

All existing original lane archives and provider identities remain immutable.
Unstarted lane identities are derived offline from explicit original job proof.
Completed lanes retire unused allocations. Started incomplete tournaments have a
60-call cap; proven provider-unstarted tournaments have a reduced 40-call cap.
Incomplete ecology lanes reserve three more calls. All allocations together must
fit the original 1,104-call ceiling. New Super repairs receive 8,192 tokens; new
missing-only sibling requests receive an explicit ID/count formatting directive.
Original payloads, rubrics, parsers, first-valid values, models and gates remain.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from village_001d_recover_completed import safe_extract, verify_zip
from village_001e_ecology_plan import build_sessions
from village_001e_provider_checkpoint import Checkpoint, json_bytes, sha256_file, sha256_value

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'experiments/village-001/qualification-20261119/config.json'
REPO, REPO_ID = 'techrote/neurounner', 1432134981
ORIGINAL_WORKFLOW = '.github/workflows/village-001e-isolated.yml'
ORIGINAL_BRANCH = 'work/village-001e-qualification-20261119'
GLOBAL_LIMIT = 1104
ECOLOGY_SUPPLEMENT = 3
STARTED_TOURNAMENT_LIMIT = 60
UNSTARTED_TOURNAMENT_LIMIT = 40
PROVIDER_STEP = 'Complete missing blind judgments within the fixed call budget'
TOKEN_POLICY = {
    'scope': 'new-logical-Super-missing-only-repairs-absent-from-original-inventory',
    'max_tokens': 8192,
    'preserve_existing_payloads': True,
    'preserve_model_messages_temperature_top_p': True,
    'private_reasoning_is_not_judgment_evidence': True,
}
FORMAT_POLICY = {
    'scope': 'new-logical-missing-only-tournament-sibling-requests-absent-from-original-inventory',
    'append_to_original_user_prompt': True,
    'enumerate_exact_requested_ids_and_count': True,
    'original_schema_example_is_illustrative': True,
    'rubric_items_model_sampling_and_scores_unchanged': True,
}
SCRIPT_BINDINGS = (
    'village_001e_recover_provider_budget.py',
    'village_001e_tournament_scenario.py', 'village_001e_ecology_judge.py',
    'village_001e_provider_checkpoint.py', 'village_001e_assemble.py',
    'village_001e_generation_checkpoint.py', 'village_001e_ecology_plan.py',
    'village_001e_generate_final_responses.py', 'village_001e_ecology_run.py',
    'village_001_manifest.py', 'village_001b_generate_member.py',
    'babble_002_expand.py', 'village_001d_recover_completed.py', 'village_001e_isolated.py',
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path: Path):
    return json.loads(path.read_text())


def write_once(path: Path, value):
    data = json_bytes(value)
    require(not path.is_symlink(), 'recovery evidence may not be a symlink')
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, 'immutable recovery evidence mismatch: ' + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tree(root: Path) -> dict:
    require(root.is_dir() and not root.is_symlink(), 'missing/nonregular recovery source directory')
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'unsafe recovery source entry')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    return result


def logical_label(label: str) -> str:
    return re.sub(r'-transport-resume-[0-9]{2}$', '', label)


def labels(root: Path) -> tuple[set[str], set[str]]:
    actual = {p.name.removesuffix('-request.json') for p in (root / 'provider').glob('*-request.json')}
    require(all(re.fullmatch(r'[A-Za-z0-9_.-]+', x) for x in actual), 'unsafe archived request label')
    return actual, {logical_label(x) for x in actual}


def expected_lanes(config: Path) -> list[dict]:
    cfg = read(config)
    rows = [{'kind': 'ecology', 'lane': x['session_id'], 'original_logical_limit': 5 if x['kind'] == 'cross_lineage_group' else 3}
            for x in build_sessions(cfg)]
    rows += [{'kind': 'tournament', 'lane': f'F{i:02d}', 'original_logical_limit': 42} for i in range(1, 25)]
    for row in rows:
        row['artifact_name'] = f"village001e-judge-{row['kind']}-{row['lane']}"
        row['terminal'] = 'judgment.json' if row['kind'] == 'ecology' else 'scenario.json'
    require(len(rows) == 48 and sum(x['original_logical_limit'] for x in rows) == GLOBAL_LIMIT, 'original campaign ceiling changed')
    return rows


def bindings(config: Path) -> dict:
    cfg = read(config)
    paths = [config, *(ROOT / 'scripts' / name for name in SCRIPT_BINDINGS)]
    paths += [ROOT / cfg[key] for key in ('campaign_config', 'final_eval_config')]
    paths += [ROOT / cfg['preference_evidence_root'] / name for name in ('summary.json', 'adapter-release.json')]
    paths += [config.parent / 'generation-custody.json']
    paths += [ROOT / ORIGINAL_WORKFLOW]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def amended_payload(label: str, payload: dict, original_labels: set[str], super_model: str) -> dict:
    # This decision uses the frozen original inventory, never the accumulating
    # recovery directory; therefore new 8192-token calls replay as 8192 forever.
    if label in original_labels:
        return payload
    result = payload
    if payload.get('model') == super_model and re.search(r'-missing-r[123]-b[0-9]{2}$', label):
        require(payload.get('max_tokens') in (2048, 6144), 'unexpected original Super repair token budget')
        result = {**payload, 'max_tokens': TOKEN_POLICY['max_tokens']}
    if re.fullmatch(r'F[0-9]{2}-(?:raw|behavior)-sibling-[0-9]{2}-missing-r[123]-b[0-9]{2}', label):
        messages = result.get('messages')
        require(isinstance(messages, list) and len(messages) == 2 and messages[1].get('role') == 'user', 'unexpected sibling message structure')
        original_prompt = messages[1]['content']
        require(isinstance(original_prompt, str) and '\nITEMS:\n' in original_prompt, 'sibling request items missing')
        items = json.loads(original_prompt.rsplit('\nITEMS:\n', 1)[1])
        require(isinstance(items, list) and items, 'missing-only sibling request has no items')
        ids = [item.get('id') for item in items]
        require(all(isinstance(x, str) and re.fullmatch(r'R[0-9]{3}', x) for x in ids) and len(ids) == len(set(ids)), 'sibling requested ID set invalid')
        clarification = ('\n\nMISSING-ONLY JSON OUTPUT REQUIREMENT (format clarification only):\n'
            + 'Return exactly ' + str(len(ids)) + ' classifications rows, one for each of these actual item IDs: '
            + json.dumps(ids) + '.\n'
            + 'The ID in the schema example is illustrative. Copy the actual IDs above. '
            + 'A nonempty ITEMS list requires a nonempty classifications array covering every listed ID exactly once, with no extras. '
            + 'Use the existing sibling rubric, confidence scale, evidence and schema unchanged. '
            + 'Other previously valid classifications are fixed and are not requested again.')
        result = {**result, 'messages': [dict(messages[0]), {**messages[1], 'content': original_prompt + clarification}]}
    return result


class FreshCall(Exception):
    def __init__(self, label: str, payload: dict, role: str):
        super().__init__('offline probe reached fresh provider work')
        self.record = {'actual_label': label, 'logical_label': logical_label(label), 'payload_sha256': sha256_value(payload),
                       'role': role, 'model': payload['model'], 'max_tokens': payload['max_tokens']}


def driver_args(config: Path, kind: str, lane: str, resume: Path | None, out: Path, offline: bool) -> list[str]:
    args = ['diagnostic-recovery', '--config', str(config), '--out', str(out)]
    if resume is not None:
        args += ['--resume', str(resume)]
    if kind == 'ecology':
        args += ['--session', str(config.parent / 'generation/ecology' / lane)]
    else:
        args += ['--scenario', lane, '--responses-root', str(config.parent / 'generation/responses')]
    return args + (['--offline'] if offline else [])


def invoke(config: Path, kind: str, lane: str, resume: Path, out: Path, checkpoint_class, *, offline: bool, quiet=False):
    module = importlib.import_module('village_001e_ecology_judge' if kind == 'ecology' else 'village_001e_tournament_scenario')
    old_checkpoint, old_argv = module.Checkpoint, sys.argv
    try:
        module.Checkpoint = checkpoint_class
        sys.argv = driver_args(config, kind, lane, resume, out, offline)
        with contextlib.ExitStack() as stack:
            if quiet:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            module.main()
    finally:
        module.Checkpoint, sys.argv = old_checkpoint, old_argv


def probe(config: Path, kind: str, lane: str, source: Path | None, *, materialize: Path | None = None) -> dict:
    if source is not None:
        tree(source)
        _, original = labels(source)
    else:
        original = set()
    require(materialize is None or source is None, 'only proven unstarted lanes may derive a zero-call source')
    campaign = read(ROOT / read(config)['campaign_config'])
    super_model = campaign['nvidia']['models']['adjudicator']['id']

    def no_provider(**kwargs):
        raise FreshCall(kwargs['label'], kwargs['payload'], kwargs['role'])

    class ProbeCheckpoint(Checkpoint):
        def __init__(self, out, **kwargs):
            # The API is replaced before any call. This is an offline execution
            # of the unchanged evaluator through its next missing request.
            kwargs.update(key='offline-probe', offline=False, api=no_provider, logical_request_limit=GLOBAL_LIMIT)
            super().__init__(out, **kwargs)

        def request(self, label, payload, role):
            return super().request(label, amended_payload(label, payload, original, super_model), role)

    with tempfile.TemporaryDirectory(prefix='village001e-budget-probe-') as directory:
        out = Path(directory) / 'replay'
        try:
            invoke(config, kind, lane, source, out, ProbeCheckpoint, offline=True, quiet=True)
        except FreshCall as pending:
            if materialize is not None:
                require(labels(out) == (set(), set()) and set(tree(out)) == {'provider-identity.json'}, 'offline unstarted derivation produced provider evidence')
                if materialize.exists():
                    require(tree(materialize) == tree(out), 'derived zero-call source changed')
                else:
                    shutil.copytree(out, materialize)
            return {'status': 'pending', 'next_unarchived_request': pending.record}
        terminal = out / ('judgment.json' if kind == 'ecology' else 'scenario.json')
        require(terminal.is_file(), 'offline probe completed without a terminal result')
        return {'status': 'completed', 'terminal_sha256': sha256_file(terminal)}


def effective_limit(row: dict) -> int:
    count = row['original_logical_count']
    if row['status'] == 'completed':
        return count
    if row['kind'] == 'tournament':
        return UNSTARTED_TOURNAMENT_LIMIT if row['status'] == 'unstarted' else STARTED_TOURNAMENT_LIMIT
    return row['original_logical_limit'] if row['status'] == 'unstarted' else count + ECOLOGY_SUPPLEMENT


def allocation(lanes: list[dict]) -> dict:
    require(sum(x['original_logical_limit'] for x in lanes) == GLOBAL_LIMIT, 'original logical allocation is not 1104')
    for row in lanes:
        count = row['original_logical_count']
        require(type(count) is int and 0 <= count <= row['original_logical_limit'], 'original lane exceeds its original cap')
        require(row['status'] in ('completed', 'pending', 'unstarted'), 'unknown recovery lane status')
        if row['status'] == 'unstarted':
            require(count == 0, 'unstarted lane has provider calls')
        expected = effective_limit(row) - count
        require(row['supplemental_logical_reservation'] == expected, 'supplemental reservation drift')
        require(row['effective_logical_limit'] == count + expected, 'effective lane cap drift')
    spent = sum(x['original_logical_count'] for x in lanes)
    supplemental = sum(x['supplemental_logical_reservation'] for x in lanes)
    require(spent + supplemental <= GLOBAL_LIMIT, 'supplemental recovery would exceed original global logical ceiling')
    return {'original_global_logical_limit': GLOBAL_LIMIT, 'original_logical_requests_archived': spent,
            'unused_original_allocations_retired': GLOBAL_LIMIT - spent,
            'supplemental_logical_requests_reserved': supplemental,
            'maximum_total_logical_requests': spent + supplemental,
            'unallocated_logical_slots': GLOBAL_LIMIT - spent - supplemental,
            'maximum_supplemental_requested_output_tokens': supplemental * TOKEN_POLICY['max_tokens'],
            'maximum_transport_launches_per_logical_request': 3,
            'allocation_policy': {'completed': 'archived logical count', 'started_incomplete_tournament': STARTED_TOURNAMENT_LIMIT,
                'proven_provider_unstarted_tournament': UNSTARTED_TOURNAMENT_LIMIT, 'incomplete_ecology_additional': ECOLOGY_SUPPLEMENT},
            'provider_unstarted_lanes': sum(row['status'] == 'unstarted' for row in lanes)}


def lane_job(jobs: list[dict], run: dict, row: dict) -> dict:
    name = f"provider ({row['kind']}, {row['lane']})"
    matches = [job for job in jobs if job.get('name') == name]
    require(len(matches) == 1, 'missing or ambiguous original matrix job: ' + name)
    job = matches[0]
    require(job.get('run_id') == run['id'] and job.get('head_sha') == run['head_sha'] and job.get('head_branch') == ORIGINAL_BRANCH,
            'original matrix job source mismatch')
    require(job.get('status') == 'completed' and job.get('run_attempt') == 1, 'original matrix job is not a completed unique attempt')
    return job


def prove_unstarted(job: dict) -> dict:
    steps = job.get('steps')
    require(isinstance(steps, list), 'original job steps are unavailable')
    matches = [step for step in steps if step.get('name') == PROVIDER_STEP]
    require(len(matches) <= 1, 'ambiguous original provider step')
    if matches:
        require(matches[0].get('status') == 'completed' and matches[0].get('conclusion') == 'skipped',
                'absent artifact has an attempted provider step; its calls cannot be assumed zero')
        basis = 'explicit-provider-step-skipped'
    else:
        require(not steps and job.get('conclusion') in ('cancelled', 'skipped') and job.get('runner_id') in (None, 0)
                and not job.get('runner_name'), 'absent artifact lacks proof that provider execution never started')
        basis = 'cancelled-or-skipped-before-runner-allocation'
    return {'kind': 'verified-zero-provider-job-proof', 'basis': basis, 'job': job,
            'provider_calls': 0, 'zip_artifact_created': False, 'identity_derivation': 'unchanged-frozen-driver-offline-no-provider-stub'}


def make_manifest(args) -> dict:
    config = args.config.resolve()
    run, inventory = read(args.run_record), read(args.artifact_inventory)
    require(run.get('status') == 'completed' and run.get('path') == ORIGINAL_WORKFLOW and run.get('head_branch') == ORIGINAL_BRANCH, 'original run must be completed on the frozen workflow/branch')
    require(all(run.get(key, {}).get('id') == REPO_ID and run[key].get('full_name') == REPO for key in ('repository', 'head_repository')), 'original run repository mismatch')
    require(type(run.get('id')) is int and re.fullmatch(r'[0-9a-f]{40}', run.get('head_sha', '')), 'invalid original run identity')
    require(run.get('run_attempt') == 1, 'original campaign must have a unique run attempt')
    jobs_document = read(args.jobs_record)
    jobs = jobs_document['jobs'] if isinstance(jobs_document, dict) else jobs_document
    require(isinstance(jobs, list), 'original jobs inventory missing')
    artifact_rows = inventory['artifacts'] if isinstance(inventory, dict) else inventory
    require(isinstance(artifact_rows, list), 'artifact inventory missing')
    by_name = {}
    for artifact in artifact_rows:
        require(artifact['name'] not in by_name, 'duplicate original artifact name')
        by_name[artifact['name']] = artifact
    result = []
    for base in expected_lanes(config):
        name = base['artifact_name']
        job = lane_job(jobs, run, base)
        source = args.artifacts / name
        artifact = by_name.get(name)
        if artifact is None:
            custody = prove_unstarted(job)
            state = probe(config, base['kind'], base['lane'], None, materialize=source)
            state['status'] = 'unstarted'
        else:
            require(not artifact.get('expired') and artifact['workflow_run']['id'] == run['id'] and artifact['workflow_run']['head_sha'] == run['head_sha'], 'original artifact provenance mismatch')
            require(re.fullmatch(r'sha256:[0-9a-f]{64}', artifact.get('digest', '')) and 0 < artifact['size_in_bytes'] < 128 * 1024 * 1024, 'original artifact digest/size invalid')
            require(artifact['archive_download_url'] == f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact['id']}/zip", 'original artifact URL drift')
            archive = args.archives / (name + '.zip')
            verify_zip(archive, artifact)
            observed = tree(source)
            with tempfile.TemporaryDirectory(prefix='village001e-budget-zip-') as directory:
                decoded = Path(directory) / 'decoded'
                safe_extract(archive, decoded)
                require(tree(decoded) == observed, 'original extracted files differ from verified ZIP')
            state = probe(config, base['kind'], base['lane'], source)
            custody = {'kind': 'verified-original-action-artifact', 'job': job}
        observed = tree(source)
        identity = read(source / 'provider-identity.json')
        require(identity.get('logical_request_limit') == base['original_logical_limit'], 'original provider identity cap drift')
        require(identity.get('session_id' if base['kind'] == 'ecology' else 'scenario_id') == base['lane'], 'original provider lane identity drift')
        actual, logical = labels(source)
        row = {**base, **state, 'original_provider_identity_sha256': sha256_file(source / 'provider-identity.json'),
            'original_files': observed, 'original_request_labels': sorted(actual), 'original_logical_labels': sorted(logical),
            'original_logical_count': len(logical), 'source_custody': custody,
            'artifact': {k: artifact[k] for k in ('id', 'name', 'digest', 'size_in_bytes', 'archive_download_url', 'workflow_run')} if artifact else None}
        row['effective_logical_limit'] = effective_limit(row)
        row['supplemental_logical_reservation'] = row['effective_logical_limit'] - len(logical)
        result.append(row)
    budget = allocation(result)
    campaign = read(ROOT / read(config)['campaign_config'])
    return {'schema': 'village-001e-provider-budget-recovery-v1', 'config': config.relative_to(ROOT).as_posix(),
            'original_run': run, 'run_record_sha256': sha256_file(args.run_record), 'artifact_inventory_sha256': sha256_file(args.artifact_inventory),
            'original_jobs': jobs, 'jobs_record_sha256': sha256_file(args.jobs_record),
            'source_sha256': bindings(config), 'lanes': result, 'budget': budget,
            'token_budget_amendment': TOKEN_POLICY, 'super_model': campaign['nvidia']['models']['adjudicator']['id'],
            'sibling_output_format_amendment': FORMAT_POLICY,
            'http_attempts_per_transport_launch': campaign['nvidia']['generation']['attempts'],
            'original_identity_and_first_valid_judgments_preserved': True,
            'original_repair_batch_sizes': [3, 2, 1], 'new_repair_rounds_added': 0,
            'new_training_or_local_generations': 0,
            'scientific_rubrics_catalogs_parsers_models_thresholds_unchanged': True,
            'provider_request_formatting_explicitly_amended': True}


def validate_manifest(path: Path) -> tuple[dict, Path]:
    manifest = read(path)
    require(manifest.get('schema') == 'village-001e-provider-budget-recovery-v1', 'wrong recovery manifest schema')
    config = (ROOT / manifest['config']).resolve()
    require(ROOT.resolve() in config.parents, 'recovery config escapes repository')
    require(manifest.get('source_sha256') == bindings(config), 'frozen recovery source bytes changed')
    expected = expected_lanes(config)
    require(len(manifest['lanes']) == len(expected), 'recovery lane population changed')
    for lane, base in zip(manifest['lanes'], expected):
        require(all(lane.get(k) == v for k, v in base.items()), 'recovery lane identity/order changed')
        require(len(lane['original_logical_labels']) == lane['original_logical_count'] and len(set(lane['original_logical_labels'])) == lane['original_logical_count'], 'original logical label count mismatch')
        require(sorted({logical_label(x) for x in lane['original_request_labels']}) == lane['original_logical_labels'], 'original transport/logical labels mismatch')
        job = lane_job(manifest['original_jobs'], manifest['original_run'], lane)
        if lane['status'] == 'unstarted':
            require(lane['artifact'] is None and lane['source_custody'] == prove_unstarted(job), 'unstarted lane lacks its original job proof')
            require(not lane['original_request_labels'] and set(lane['original_files']) == {'provider-identity.json'}, 'unstarted lane contains fabricated provider evidence')
        else:
            require(lane['artifact'] is not None and lane['source_custody'] == {'kind': 'verified-original-action-artifact', 'job': job}, 'started lane lacks original artifact custody')
    require(manifest.get('budget') == allocation(manifest['lanes']), 'recovery accounting changed')
    require(manifest.get('token_budget_amendment') == TOKEN_POLICY and manifest.get('sibling_output_format_amendment') == FORMAT_POLICY and manifest.get('original_repair_batch_sizes') == [3, 2, 1] and manifest.get('new_repair_rounds_added') == 0, 'recovery semantics/policy changed')
    campaign = read(ROOT / read(config)['campaign_config'])
    require(manifest.get('super_model') == campaign['nvidia']['models']['adjudicator']['id'], 'recovery adjudicator changed')
    require(manifest.get('http_attempts_per_transport_launch') == campaign['nvidia']['generation']['attempts'], 'transport attempt budget changed')
    require(manifest['original_run'].get('status') == 'completed', 'original run has not completed')
    return manifest, config


def lane_record(manifest: dict, kind: str, lane: str) -> dict:
    matches = [x for x in manifest['lanes'] if x['kind'] == kind and x['lane'] == lane]
    require(len(matches) == 1, 'lane is outside the explicit recovery manifest')
    return matches[0]


def preserve_original(root: Path, row: dict) -> dict:
    observed = tree(root)
    require(all(observed.get(name) == value for name, value in row['original_files'].items()), 'original provider archive bytes changed or disappeared')
    require(sha256_file(root / 'provider-identity.json') == row['original_provider_identity_sha256'], 'original provider identity was rewritten')
    actual, logical = labels(root)
    require(set(row['original_request_labels']) <= actual and set(row['original_logical_labels']) <= logical, 'original request history lost')
    require(len(logical) <= row['effective_logical_limit'], 'effective lane recovery cap exceeded')
    if actual - set(row['original_request_labels']):
        require((root / 'budget-recovery-identity.json').is_file(), 'supplemental provider history lacks its recovery identity')
    return observed


def recovery_identity(path: Path, manifest: dict, row: dict) -> dict:
    return {'schema': 'village-001e-budget-recovery-execution-identity-v1', 'manifest_sha256': sha256_file(path),
            'wrapper_sha256': sha256_file(Path(__file__)), 'kind': row['kind'], 'lane': row['lane'],
            'original_provider_identity_sha256': row['original_provider_identity_sha256'],
            'original_artifact_sha256': row['artifact']['digest'] if row['artifact'] else None,
            'original_source_custody_sha256': sha256_value(row.get('source_custody')),
            'original_lane_state': row['status'],
            'original_logical_limit': row['original_logical_limit'],
            'original_logical_requests_archived': row['original_logical_count'],
            'supplemental_logical_reservation': row['supplemental_logical_reservation'],
            'effective_logical_limit': row['effective_logical_limit'],
            'original_global_logical_limit': GLOBAL_LIMIT,
            'maximum_campaign_logical_requests': manifest['budget']['maximum_total_logical_requests'],
            'token_budget_amendment': TOKEN_POLICY, 'sibling_output_format_amendment': FORMAT_POLICY,
            'original_provider_identity_preserved': True}


def checkpoint_factory(row: dict, manifest: dict, identity: dict, *, offline: bool, capture: list):
    original = set(row['original_logical_labels'])

    class RecoveryCheckpoint(Checkpoint):
        def __init__(self, out, **kwargs):
            require(kwargs['logical_request_limit'] == row['original_logical_limit'], 'frozen driver changed its declared original cap')
            kwargs['logical_request_limit'] = row['effective_logical_limit']
            kwargs['offline'] = offline
            super().__init__(out, **kwargs)
            self.first_pending_seen = False
            write_once(out / 'budget-recovery-identity.json', identity)
            preserve_original(out, row)
            capture.append(self)

        def request(self, label, payload, role):
            payload = amended_payload(label, payload, original, manifest['super_model'])
            expected = row.get('next_unarchived_request')
            if expected is not None and logical_label(label) == expected['logical_label']:
                require(sha256_value(payload) == expected['payload_sha256'] and role == expected['role'], 'first supplemental request differs from frozen offline probe')
                self.first_pending_seen = True
            elif label not in original:
                require(self.first_pending_seen, 'supplemental work preceded the frozen offline probe boundary')
            return super().request(label, payload, role)

    return RecoveryCheckpoint


def run_lane(path: Path, manifest: dict, config: Path, row: dict, source: Path, out: Path, *, offline: bool, quiet=False):
    preserve_original(source, row)
    identity = recovery_identity(path, manifest, row)
    existing_identity = source / 'budget-recovery-identity.json'
    if existing_identity.exists():
        require(read(existing_identity) == identity, 'recovery resume belongs to another manifest or wrapper')
    if not offline:
        require(os.environ.get('GITHUB_RUN_ATTEMPT', '1') == '1', 'live recovery reruns require an explicit fresh request with preserved prior raw artifacts')
    initial_actual, initial_logical = labels(source)
    capture, failure = [], None
    cls = checkpoint_factory(row, manifest, identity, offline=offline, capture=capture)
    try:
        invoke(config, row['kind'], row['lane'], source, out, cls, offline=offline, quiet=quiet)
        terminal = out / row['terminal']
        require(terminal.is_file(), 'recovery did not produce terminal judgment')
        if row['status'] == 'completed':
            require(sha256_file(terminal) == row['terminal_sha256'], 'completed original judgment changed')
    except Exception as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        if capture:
            preserve_original(out, row)
            actual, logical = labels(out)
            receipt = {'schema': 'village-001e-budget-recovery-invocation-v1', 'manifest_sha256': sha256_file(path),
                'offline': offline, 'status': 'failed' if failure else 'completed', 'error': failure,
                'new_transport_launches': len(actual - initial_actual), 'new_logical_requests': len(logical - initial_logical),
                'logical_requests_archived': len(logical), 'original_logical_requests_archived': row['original_logical_count'],
                'effective_logical_limit': row['effective_logical_limit'], 'new_provider_calls': capture[0].new_calls,
                'original_files_preserved': True}
            index = 1
            while (out / f'budget-recovery-execution-{index:03d}.json').exists():
                index += 1
            write_once(out / f'budget-recovery-execution-{index:03d}.json', receipt)


def verify_complete(path: Path, manifest: dict, config: Path, artifacts: Path) -> dict:
    expected = {row['artifact_name'] for row in manifest['lanes']}
    require({p.name for p in artifacts.glob('village001e-judge-*')} == expected, 'final recovery artifact set is not exactly 48 lanes')
    results = []
    with tempfile.TemporaryDirectory(prefix='village001e-final-budget-replay-') as directory:
        for row in manifest['lanes']:
            source = artifacts / row['artifact_name']
            preserve_original(source, row)
            require(read(source / 'budget-recovery-identity.json') == recovery_identity(path, manifest, row), 'final recovery identity missing or changed')
            run_lane(path, manifest, config, row, source, Path(directory) / row['artifact_name'], offline=True, quiet=True)
            actual, logical = labels(source)
            amended = []
            for label in sorted(logical - set(row['original_logical_labels'])):
                request = read(source / 'provider' / (label + '-request.json'))
                if request['payload']['model'] == manifest['super_model'] and re.search(r'-missing-r[123]-b[0-9]{2}$', label):
                    require(request['payload']['max_tokens'] == 8192, 'new Super repair has wrong amended budget')
                    amended.append(label)
            results.append({'kind': row['kind'], 'lane': row['lane'], 'original_logical_requests': row['original_logical_count'],
                'logical_requests': len(logical), 'transport_launches': len(actual), 'supplemental_logical_requests': len(logical) - row['original_logical_count'],
                'effective_logical_limit': row['effective_logical_limit'], 'amended_super_repair_labels': amended,
                'terminal_sha256': sha256_file(source / row['terminal']), 'artifact_tree_sha256': sha256_value(tree(source)),
                'offline_replay_passed': True, 'original_files_preserved': True})
    total = sum(x['logical_requests'] for x in results)
    require(total <= manifest['budget']['maximum_total_logical_requests'] <= GLOBAL_LIMIT, 'final global logical request ceiling exceeded')
    return {'schema': 'village-001e-global-provider-recovery-accounting-v1', 'manifest_sha256': sha256_file(path),
            'budget': manifest['budget'], 'actual_logical_requests': total,
            'actual_transport_launches': sum(x['transport_launches'] for x in results),
            'actual_supplemental_logical_requests': sum(x['supplemental_logical_requests'] for x in results),
            'all_48_lanes_reconstructed_offline': True, 'original_identities_and_judgments_preserved': True,
            'token_budget_amendment': TOKEN_POLICY, 'sibling_output_format_amendment': FORMAT_POLICY,
            'scientific_gates_unchanged': True, 'lanes': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('manifest')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    for name in ('artifacts', 'archives', 'run-record', 'artifact-inventory', 'jobs-record', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p = sub.add_parser('probe')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--kind', choices=('ecology', 'tournament'), required=True)
    p.add_argument('--lane', required=True)
    p.add_argument('--resume', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--kind', choices=('ecology', 'tournament'), required=True)
    p.add_argument('--lane', required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--offline', action='store_true')
    p = sub.add_parser('verify-complete')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'manifest':
        result = make_manifest(args)
        write_once(args.out, result)
        print(json.dumps(result['budget'], indent=2))
    elif args.command == 'probe':
        print(json.dumps(probe(args.config.resolve(), args.kind, args.lane, args.resume), indent=2))
    else:
        manifest, config = validate_manifest(args.manifest)
        if args.command == 'run':
            run_lane(args.manifest, manifest, config, lane_record(manifest, args.kind, args.lane), args.resume, args.out, offline=args.offline)
        else:
            result = verify_complete(args.manifest, manifest, config, args.artifacts)
            write_once(args.out, result)
            print(json.dumps({key: result[key] for key in ('actual_logical_requests', 'actual_supplemental_logical_requests', 'all_48_lanes_reconstructed_offline')}, indent=2))


if __name__ == '__main__':
    main()
