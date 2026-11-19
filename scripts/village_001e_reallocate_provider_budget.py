#!/usr/bin/env python3
"""Source-bound Stage E allocation extension; frozen evaluators remain unchanged.

The verified first recovery snapshot is the accounting boundary. Completed lanes
retire their unused reservations; incomplete lanes receive 60 logical calls in
total, except F15's explicitly frozen 69-call structural repair envelope. The
whole allocation must fit the original 1,104-call campaign ceiling. Existing
provider requests, judgments, original identities and recovery identities are
never rewritten. No repair rounds, scoring rules, models or prompts are added.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import village_001e_recover_provider_budget as frozen
from village_001e_provider_checkpoint import json_bytes, sha256_file, sha256_value

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'experiments/village-001/qualification-20261119/provider-recovery/manifest.json'
ORIGINAL_MANIFEST_SHA256 = '805a0fc80513c291c2906e0a12115278d2b7f87c30544725c08dcfc4d75d96d6'
BASELINE_RUN = 44443194861
BASELINE_HEAD = 'abd44e3c21a9bcd63cef1f9310df34e05806f40b'
BASELINE_ARTIFACT = 14498971739
BASELINE_ZIP_SHA256 = '4c5f263a836e8a38fc7e8cdeb0143103072745eb8a628d569ebc2eda2727d70d'
BASELINE_ZIP_BYTES = 5208218
RECOVERY_WORKFLOW = '.github/workflows/village-001e-recovery.yml'
IDENTITY = 'allocation-extension-identity.json'
SCHEMA = 'village-001e-provider-allocation-extension-v1'
POLICY = {
    'completed': 'exact archived logical count after successful frozen offline replay',
    'incomplete_tournament_logical_limit': 60,
    'structural_exceptions': {'F15': {
        'logical_limit': 69, 'baseline_logical_count': 40,
        'maximum_remaining_original_schedule_requests': 29,
        'basis': 'current three-row missing batch: 1+2+3; final twelve-row batch: 1+4+6+12',
        'next_logical_label': 'F15-behavior-sibling-02-missing-r1-b04',
    }},
    'selection_uses_completion_and_remaining_schedule_only': True,
    'scientific_scores_not_used_for_allocation': True,
    'repair_batch_sizes': [3, 2, 1], 'new_repair_rounds_added': 0,
}

require = frozen.require
read = frozen.read
write_once = frozen.write_once
tree = frozen.tree
labels = frozen.labels


def original_path(allocation: dict) -> Path:
    path = (ROOT / allocation['original_manifest']['path']).resolve()
    require(ROOT.resolve() in path.parents, 'original manifest path escapes repository')
    return path


def lane_record(allocation: dict, kind: str, lane: str) -> dict:
    return frozen.lane_record(allocation, kind, lane)


def effective_limit(row: dict) -> int:
    if row['baseline_status'] == 'completed':
        return row['baseline_logical_count']
    require(row['kind'] == 'tournament', 'this extension does not authorize incomplete ecology')
    exception = POLICY['structural_exceptions'].get(row['lane'])
    if exception:
        require(row['baseline_logical_count'] == exception['baseline_logical_count'], 'structural exception baseline count changed')
        require(row['next_unarchived_request']['logical_label'] == exception['next_logical_label'], 'structural exception pending boundary changed')
        return exception['logical_limit']
    return POLICY['incomplete_tournament_logical_limit']


def allocation_budget(rows: list[dict]) -> dict:
    for row in rows:
        count = row['baseline_logical_count']
        require(type(count) is int and count >= 0, 'invalid baseline logical count')
        require(row['baseline_status'] in ('completed', 'pending'), 'unknown baseline completeness')
        limit = effective_limit(row)
        require(type(row['effective_logical_limit']) is int and row['effective_logical_limit'] == limit and count <= limit,
                'allocation cap or baseline spending changed')
        require(row['remaining_logical_reservation'] == limit - count, 'remaining logical reservation changed')
    spent = sum(row['baseline_logical_count'] for row in rows)
    total = sum(row['effective_logical_limit'] for row in rows)
    require(total <= frozen.GLOBAL_LIMIT, 'allocation extension exceeds original global logical ceiling')
    return {
        'original_global_logical_limit': frozen.GLOBAL_LIMIT,
        'baseline_logical_requests_archived': spent,
        'baseline_transport_launches_archived': sum(len(row['baseline_actual_labels']) for row in rows),
        'completed_lanes': sum(row['baseline_status'] == 'completed' for row in rows),
        'completed_lane_logical_requests': sum(row['baseline_logical_count'] for row in rows if row['baseline_status'] == 'completed'),
        'incomplete_lanes': sum(row['baseline_status'] == 'pending' for row in rows),
        'remaining_logical_requests_reserved': total - spent,
        'maximum_total_logical_requests': total,
        'unallocated_logical_slots': frozen.GLOBAL_LIMIT - total,
        'maximum_transport_launches_per_logical_request': 3,
    }


def old_identity(allocation: dict, original: dict, row: dict) -> dict:
    old = lane_record(original, row['kind'], row['lane'])
    return frozen.recovery_identity(original_path(allocation), original, old)


def extension_identity(allocation: dict, row: dict) -> dict:
    return {
        'schema': 'village-001e-allocation-extension-identity-v1',
        'allocation_manifest_sha256': sha256_value(allocation),
        'allocation_module_sha256': sha256_file(Path(__file__)),
        'original_manifest_sha256': allocation['original_manifest']['sha256'],
        'baseline_run_id': allocation['prior_recovery_run']['id'],
        'baseline_artifact_sha256': allocation['prior_recovery_artifact']['digest'],
        'baseline_source_tree_sha256': sha256_value(row['baseline_files']),
        'kind': row['kind'], 'lane': row['lane'],
        'baseline_logical_requests': row['baseline_logical_count'],
        'previous_effective_logical_limit': row['previous_effective_logical_limit'],
        'effective_logical_limit': row['effective_logical_limit'],
        'maximum_campaign_logical_requests': allocation['budget']['maximum_total_logical_requests'],
        'original_global_logical_limit': frozen.GLOBAL_LIMIT,
        'original_provider_and_recovery_identities_preserved': True,
        'existing_payloads_and_first_valid_judgments_preserved': True,
    }


def validate_prior_source(source: Path, allocation: dict, row: dict, *, original: dict | None = None) -> dict:
    require(lane_record(allocation, row['kind'], row['lane']) == row, 'lane does not match active allocation')
    if original is None:
        original, _ = frozen.validate_manifest(original_path(allocation))
    old = lane_record(original, row['kind'], row['lane'])
    adjusted = {**old, 'effective_logical_limit': row['effective_logical_limit']}
    observed = frozen.preserve_original(source, adjusted)
    require(all(observed.get(name) == value for name, value in row['baseline_files'].items()),
            'preserved recovery snapshot bytes changed or disappeared; refusing budget reset')
    actual, logical = labels(source)
    require(set(row['baseline_actual_labels']) <= actual and set(row['baseline_logical_labels']) <= logical,
            'baseline provider request history lost; refusing budget reset')
    existing_old = source / 'budget-recovery-identity.json'
    if existing_old.exists():
        require(read(existing_old) == old_identity(allocation, original, row), 'original recovery identity changed')
    existing = source / IDENTITY
    if actual - set(row['baseline_actual_labels']):
        require(existing.is_file(), 'post-baseline provider history lacks allocation extension identity')
    if existing.exists():
        require(read(existing) == extension_identity(allocation, row), 'resume belongs to another allocation extension')
    pending = row.get('next_unarchived_request')
    if pending and actual - set(row['baseline_actual_labels']):
        require(pending['actual_label'] in actual, 'extension history skipped the frozen pending boundary')
        request = read(source / 'provider' / (pending['actual_label'] + '-request.json'))
        require(sha256_value(request['payload']) == pending['payload_sha256'] and request['role'] == pending['role'],
                'archived allocation boundary payload changed')
    if row['baseline_status'] == 'completed':
        require(actual == set(row['baseline_actual_labels']), 'completed lane spent a new transport launch')
    return observed


def probe_baseline(path: Path, original: dict, config: Path, old: dict, source: Path) -> dict:
    """Replay using the original inventory, with an API stub before any I/O."""
    frozen.preserve_original(source, old)
    identity = frozen.recovery_identity(path, original, old)
    prior_identity = source / 'budget-recovery-identity.json'
    if prior_identity.exists():
        require(read(prior_identity) == identity, 'baseline recovery identity changed')
    # Temporarily lifting the local cap is only for finding the next request.
    # The stub raises before any provider request file or network call is made.
    adjusted = {**old, 'effective_logical_limit': frozen.GLOBAL_LIMIT}
    base = frozen.checkpoint_factory(adjusted, original, identity, offline=False, capture=[])

    def no_provider(**kwargs):
        raise frozen.FreshCall(kwargs['label'], kwargs['payload'], kwargs['role'])

    class Probe(base):
        def __init__(self, out, **kwargs):
            kwargs.update(key='offline-allocation-probe', api=no_provider)
            super().__init__(out, **kwargs)

    with tempfile.TemporaryDirectory(prefix='village001e-allocation-probe-') as directory:
        out = Path(directory) / 'lane'
        try:
            frozen.invoke(config, old['kind'], old['lane'], source, out, Probe, offline=True, quiet=True)
        except frozen.FreshCall as pending:
            return {'baseline_status': 'pending', 'next_unarchived_request': pending.record}
        terminal = out / old['terminal']
        require(terminal.is_file(), 'baseline offline replay produced no terminal judgment')
        return {'baseline_status': 'completed', 'baseline_terminal_sha256': sha256_file(terminal)}


def _metadata(path: Path, value) -> dict:
    resolved = path.resolve()
    relative = resolved.relative_to(ROOT).as_posix() if ROOT in resolved.parents else None
    return {'path': relative, 'sha256': sha256_file(path), 'value': value}


def validate_baseline_metadata(run: dict, artifact: dict, jobs: dict) -> None:
    expected = {'id': BASELINE_RUN, 'head_sha': BASELINE_HEAD, 'head_branch': frozen.ORIGINAL_BRANCH,
                'path': RECOVERY_WORKFLOW, 'run_attempt': 1, 'event': 'push', 'status': 'completed'}
    require(all(run.get(key) == value for key, value in expected.items()), 'baseline recovery run identity changed')
    require(all(run.get(key, {}).get('id') == frozen.REPO_ID and run[key].get('full_name') == frozen.REPO
                for key in ('repository', 'head_repository')), 'baseline repository changed')
    require(artifact.get('id') == BASELINE_ARTIFACT and artifact.get('name') == f'village001e-budget-recovery-{BASELINE_RUN}'
            and artifact.get('digest') == 'sha256:' + BASELINE_ZIP_SHA256 and artifact.get('size_in_bytes') == BASELINE_ZIP_BYTES
            and artifact.get('expired') is False, 'baseline recovery artifact identity changed')
    require(artifact.get('archive_download_url') == f'https://api.github.com/repos/{frozen.REPO}/actions/artifacts/{BASELINE_ARTIFACT}/zip',
            'baseline artifact download URL changed')
    require(all(artifact.get('workflow_run', {}).get(key) == value for key, value in {
        'id': BASELINE_RUN, 'head_sha': BASELINE_HEAD, 'head_branch': frozen.ORIGINAL_BRANCH,
        'repository_id': frozen.REPO_ID, 'head_repository_id': frozen.REPO_ID}.items()), 'baseline artifact run binding changed')
    require(type(jobs.get('total_count')) is int and jobs['total_count'] == len(jobs.get('jobs', [])) <= 100,
            'complete bounded baseline job inventory required')
    selected = [job for job in jobs['jobs'] if job.get('name') == 'recover']
    require(len(selected) == 1, 'baseline recovery job missing or ambiguous')
    job = selected[0]
    require(all(job.get(key) == value for key, value in {'id': 135289021869, 'run_id': BASELINE_RUN,
        'head_sha': BASELINE_HEAD, 'run_attempt': 1, 'status': 'completed'}.items()), 'baseline recovery job changed')
    steps = [step for step in job.get('steps', []) if step.get('number') == 5]
    require(len(steps) == 1 and steps[0].get('name') == 'Complete only missing provider fields within the frozen global ceiling'
            and steps[0].get('status') == 'completed' and steps[0].get('conclusion') != 'skipped',
            'baseline provider execution proof missing')


def make_manifest(args) -> dict:
    path = args.manifest.resolve()
    require(sha256_file(path) == ORIGINAL_MANIFEST_SHA256, 'original allocation manifest changed')
    original, config = frozen.validate_manifest(path)
    run, jobs, inventory = read(args.run_record), read(args.jobs_record), read(args.artifact_inventory)
    require(type(inventory.get('total_count')) is int and inventory['total_count'] == len(inventory.get('artifacts', [])) <= 100,
            'complete bounded baseline artifact inventory required')
    candidates = [a for a in inventory['artifacts'] if a.get('id') == BASELINE_ARTIFACT]
    require(len(candidates) == 1, 'baseline recovery artifact absent or duplicated')
    artifact = candidates[0]
    validate_baseline_metadata(run, artifact, jobs)
    frozen.verify_zip(args.prior_zip, artifact)
    observed = tree(args.prior_root)
    with tempfile.TemporaryDirectory(prefix='village001e-allocation-zip-') as directory:
        decoded = Path(directory) / 'decoded'
        frozen.safe_extract(args.prior_zip, decoded)
        require(tree(decoded) == observed, 'baseline directory differs from the verified original ZIP')
    require({p.name for p in args.prior_root.iterdir()} <= {'inherited', 'lanes', 'custody', 'recovery-failures.json'},
            'unexpected baseline recovery entry')
    names = {row['artifact_name'] for row in original['lanes']}
    require({p.name for p in (args.prior_root / 'inherited').iterdir()} == names, 'baseline inherited forest must contain all 48 lanes')
    require({p.name for p in (args.prior_root / 'lanes').iterdir()} <= names, 'unexpected baseline output lane')
    rows = []
    for old in original['lanes']:
        name = old['artifact_name']
        inherited = args.prior_root / 'inherited' / name
        frozen.preserve_original(inherited, old)
        selected = args.prior_root / 'lanes' / name
        if not selected.exists():
            selected = inherited
        files = tree(selected)
        require(all(files.get(key) == value for key, value in tree(inherited).items()), 'baseline output lost inherited evidence')
        require(not (selected / IDENTITY).exists(), 'baseline already contains allocation-extension execution')
        state = probe_baseline(path, original, config, old, selected)
        actual, logical = labels(selected)
        row = {key: old[key] for key in ('kind', 'lane', 'artifact_name', 'terminal', 'status')}
        row.update(state)
        row.update({'baseline_source_path': selected.relative_to(args.prior_root).as_posix(),
            'baseline_files': files, 'baseline_source_tree_sha256': sha256_value(files),
            'baseline_actual_labels': sorted(actual), 'baseline_logical_labels': sorted(logical),
            'baseline_logical_count': len(logical), 'previous_effective_logical_limit': old['effective_logical_limit'],
            'original_provider_identity_sha256': old['original_provider_identity_sha256'],
            'original_recovery_identity_sha256': sha256_value(frozen.recovery_identity(path, original, old))})
        row['effective_logical_limit'] = effective_limit(row)
        row['remaining_logical_reservation'] = row['effective_logical_limit'] - len(logical)
        rows.append(row)
    budget = allocation_budget(rows)
    require((budget['completed_lanes'], budget['completed_lane_logical_requests'], budget['baseline_logical_requests_archived'],
             budget['baseline_transport_launches_archived'], budget['maximum_total_logical_requests']) == (41, 630, 673, 674, 1059),
            'verified baseline completeness or structural allocation differs from approved extension')
    return {'schema': SCHEMA, 'original_manifest': {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256_file(path)},
        'source_sha256': {'scripts/' + Path(__file__).name: sha256_file(Path(__file__))},
        'prior_recovery_run': run, 'prior_recovery_artifact': artifact,
        'metadata': {'run': _metadata(args.run_record, run), 'jobs': _metadata(args.jobs_record, jobs),
                     'artifacts': _metadata(args.artifact_inventory, inventory)},
        'prior_bundle_tree_sha256': sha256_value(observed), 'prior_bundle_file_count': len(observed),
        'prior_bundle_file_bytes': sum(value['bytes'] for value in observed.values()),
        'policy': POLICY, 'lanes': rows, 'budget': budget,
        'token_budget_amendment': frozen.TOKEN_POLICY, 'sibling_output_format_amendment': frozen.FORMAT_POLICY,
        'http_attempts_per_transport_launch': original['http_attempts_per_transport_launch'],
        'original_provider_and_recovery_identities_preserved': True,
        'existing_payloads_and_first_valid_judgments_preserved': True,
        'scientific_scripts_models_rubrics_parsers_thresholds_unchanged': True,
        'new_training_or_local_generations': 0, 'new_repair_rounds_added': 0}


def load_allocation(path: Path) -> tuple[dict, dict, Path]:
    allocation = read(path)
    require(path.read_bytes() == json_bytes(allocation), 'allocation manifest must use its canonical immutable encoding')
    require(allocation.get('schema') == SCHEMA and allocation.get('policy') == POLICY, 'allocation schema or policy changed')
    require(allocation.get('source_sha256') == {'scripts/' + Path(__file__).name: sha256_file(Path(__file__))}, 'allocation module source changed')
    prior = original_path(allocation)
    require(allocation['original_manifest']['sha256'] == ORIGINAL_MANIFEST_SHA256 and sha256_file(prior) == ORIGINAL_MANIFEST_SHA256,
            'original manifest changed under allocation extension')
    original, config = frozen.validate_manifest(prior)
    metadata = allocation['metadata']
    for record in metadata.values():
        require(re.fullmatch(r'[0-9a-f]{64}', record['sha256']), 'invalid metadata source hash')
        if record['path'] is not None:
            source = (ROOT / record['path']).resolve()
            require(ROOT in source.parents and sha256_file(source) == record['sha256'] and read(source) == record['value'],
                    'allocation metadata source bytes changed')
    require(metadata['run']['value'] == allocation['prior_recovery_run'], 'baseline run record mismatch')
    inventory = metadata['artifacts']['value']
    require(type(inventory.get('total_count')) is int and inventory['total_count'] == len(inventory.get('artifacts', [])) <= 100,
            'baseline artifact inventory changed')
    matches = [a for a in inventory['artifacts'] if a.get('id') == BASELINE_ARTIFACT]
    require(matches == [allocation['prior_recovery_artifact']], 'baseline artifact record mismatch')
    validate_baseline_metadata(allocation['prior_recovery_run'], allocation['prior_recovery_artifact'], metadata['jobs']['value'])
    require(len(allocation['lanes']) == len(original['lanes']), 'allocation lane population changed')
    for row, old in zip(allocation['lanes'], original['lanes']):
        require(all(row.get(key) == old[key] for key in ('kind', 'lane', 'artifact_name', 'terminal', 'status')), 'allocation lane identity or order changed')
        require(row['previous_effective_logical_limit'] == old['effective_logical_limit'], 'previous lane cap changed')
        require(row['original_provider_identity_sha256'] == old['original_provider_identity_sha256']
                and row['original_recovery_identity_sha256'] == sha256_value(frozen.recovery_identity(prior, original, old)),
                'allocation changed an original identity')
        files = row['baseline_files']
        require(row['baseline_source_tree_sha256'] == sha256_value(files), 'baseline source tree digest changed')
        require(all(files.get(key) == value for key, value in old['original_files'].items()), 'baseline omitted original evidence')
        require(row['baseline_source_path'] in ('lanes/' + row['artifact_name'], 'inherited/' + row['artifact_name']), 'unsafe baseline lane selection')
        actual = row['baseline_actual_labels']
        require(actual == sorted(set(actual)) and all(re.fullmatch(r'[A-Za-z0-9_.-]+', label) for label in actual), 'invalid baseline request labels')
        require(sorted({frozen.logical_label(label) for label in actual}) == row['baseline_logical_labels']
                and len(row['baseline_logical_labels']) == row['baseline_logical_count'], 'baseline label accounting changed')
        require(set(old['original_request_labels']) <= set(actual), 'baseline original request history lost')
        require({name.removeprefix('provider/').removesuffix('-request.json') for name in files if name.startswith('provider/') and name.endswith('-request.json')} == set(actual),
                'baseline files/request inventory mismatch')
        require(row['baseline_logical_count'] <= old['effective_logical_limit'], 'baseline exceeded its previous allocation')
        if row['baseline_status'] == 'completed':
            require(re.fullmatch(r'[0-9a-f]{64}', row['baseline_terminal_sha256']) and 'next_unarchived_request' not in row,
                    'completed baseline lacks a terminal reconstruction hash')
            if row['terminal'] in files:
                require(files[row['terminal']]['sha256'] == row['baseline_terminal_sha256'], 'completed baseline terminal changed')
        else:
            require('next_unarchived_request' in row and 'baseline_terminal_sha256' not in row, 'incomplete baseline lacks its frozen pending boundary')
    require(allocation['budget'] == allocation_budget(allocation['lanes']), 'allocation accounting changed')
    require(allocation['token_budget_amendment'] == frozen.TOKEN_POLICY and allocation['sibling_output_format_amendment'] == frozen.FORMAT_POLICY
            and allocation['http_attempts_per_transport_launch'] == original['http_attempts_per_transport_launch']
            and allocation['new_training_or_local_generations'] == 0 and allocation['new_repair_rounds_added'] == 0,
            'allocation extension changed evaluation or transport policy')
    return allocation, original, config


def validate_baseline_bundle(source: Path, allocation: dict) -> None:
    observed = tree(source)
    require(sha256_value(observed) == allocation['prior_bundle_tree_sha256'] and len(observed) == allocation['prior_bundle_file_count']
            and sum(row['bytes'] for row in observed.values()) == allocation['prior_bundle_file_bytes'],
            'first allocation activation did not preserve the exact frozen recovery bundle')
    for row in allocation['lanes']:
        require(tree(source / row['baseline_source_path']) == row['baseline_files'], 'selected baseline lane bytes changed')


def checkpoint_factory(allocation: dict, original: dict, row: dict, *, offline: bool, capture: list):
    old = lane_record(original, row['kind'], row['lane'])
    adjusted = {**old, 'effective_logical_limit': row['effective_logical_limit']}
    base = frozen.checkpoint_factory(adjusted, original, old_identity(allocation, original, row), offline=offline, capture=capture)

    class Extension(base):
        def __init__(self, out, **kwargs):
            api = kwargs['api']
            pending = row.get('next_unarchived_request')
            actual, _ = labels(out)
            self.extension_boundary_seen = bool(pending and pending['actual_label'] in actual)

            def bounded_api(**call):
                if not self.extension_boundary_seen:
                    require(pending is not None and call['label'] == pending['actual_label']
                            and sha256_value(call['payload']) == pending['payload_sha256'] and call['role'] == pending['role'],
                            'new work differs from frozen allocation pending boundary')
                    self.extension_boundary_seen = True
                return api(**call)

            kwargs['api'] = bounded_api
            super().__init__(out, **kwargs)
            write_once(out / IDENTITY, extension_identity(allocation, row))
            validate_prior_source(out, allocation, row, original=original)

    return Extension


def run_lane(path: Path, allocation: dict, original: dict, config: Path, row: dict,
             source: Path, out: Path, *, offline: bool, quiet=False):
    require(sha256_file(path) == sha256_value(allocation), 'active allocation bytes changed')
    initial_files = validate_prior_source(source, allocation, row, original=original)
    initial_actual, initial_logical = labels(source)
    if row['baseline_status'] == 'completed':
        require(offline, 'completed baseline lanes are replay-only')
    if not offline:
        require(os.environ.get('GITHUB_RUN_ATTEMPT', '1') == '1', 'live allocation reruns require a fresh request with preserved latest custody')
    if not quiet:
        print(f"allocation lane {row['kind']}/{row['lane']}: {len(initial_logical)}/{row['effective_logical_limit']} logical requests archived", file=sys.stderr, flush=True)
    capture, failure, status, terminal_verified = [], None, 'failed', False
    cls = checkpoint_factory(allocation, original, row, offline=offline, capture=capture)
    try:
        frozen.invoke(config, row['kind'], row['lane'], source, out, cls, offline=offline, quiet=quiet)
        terminal = out / row['terminal']
        require(terminal.is_file(), 'allocation execution did not produce a terminal judgment')
        if row['baseline_status'] == 'completed':
            require(sha256_file(terminal) == row['baseline_terminal_sha256'], 'completed baseline judgment changed')
        terminal_verified, status = True, 'completed'
    except BaseException as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
        status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        raise
    finally:
        if capture:
            current = validate_prior_source(out, allocation, row, original=original)
            require(all(current.get(name) == value for name, value in initial_files.items()), 'allocation invocation rewrote or lost resume evidence')
            actual, logical = labels(out)
            receipt = {'schema': 'village-001e-allocation-extension-invocation-v1',
                'allocation_manifest_sha256': sha256_file(path), 'offline': offline, 'status': status, 'error': failure,
                'source_tree_sha256': sha256_value(initial_files), 'terminal_verified': terminal_verified,
                'new_transport_launches': len(actual - initial_actual), 'new_logical_requests': len(logical - initial_logical),
                'logical_requests_archived': len(logical), 'transport_launches_archived': len(actual),
                'baseline_logical_requests_archived': row['baseline_logical_count'],
                'effective_logical_limit': row['effective_logical_limit'], 'new_provider_calls_attempted': capture[0].new_calls,
                'previous_source_files_preserved': True}
            index = 1
            while (out / f'allocation-extension-execution-{index:03d}.json').exists():
                index += 1
            write_once(out / f'allocation-extension-execution-{index:03d}.json', receipt)


def verify_complete(path: Path, allocation: dict, original: dict, config: Path, artifacts: Path) -> dict:
    expected = {row['artifact_name'] for row in allocation['lanes']}
    require({p.name for p in artifacts.iterdir()} == expected, 'final allocation output is not exactly all 48 lanes')
    results = []
    with tempfile.TemporaryDirectory(prefix='village001e-allocation-final-replay-') as directory:
        for row in allocation['lanes']:
            source = artifacts / row['artifact_name']
            validate_prior_source(source, allocation, row, original=original)
            require(read(source / IDENTITY) == extension_identity(allocation, row), 'final allocation identity missing or changed')
            run_lane(path, allocation, original, config, row, source, Path(directory) / row['artifact_name'], offline=True, quiet=True)
            actual, logical = labels(source)
            old = lane_record(original, row['kind'], row['lane'])
            results.append({'kind': row['kind'], 'lane': row['lane'],
                'original_logical_requests': old['original_logical_count'],
                'baseline_logical_requests': row['baseline_logical_count'],
                'logical_requests': len(logical), 'transport_launches': len(actual),
                'supplemental_logical_requests': len(logical) - old['original_logical_count'],
                'post_allocation_logical_requests': len(logical) - row['baseline_logical_count'],
                'effective_logical_limit': row['effective_logical_limit'],
                'terminal_sha256': sha256_file(source / row['terminal']), 'artifact_tree_sha256': sha256_value(tree(source)),
                'offline_replay_passed': True, 'original_files_preserved': True, 'baseline_files_preserved': True})
    total = sum(row['logical_requests'] for row in results)
    require(total <= allocation['budget']['maximum_total_logical_requests'] <= frozen.GLOBAL_LIMIT, 'final allocation exceeded campaign ceiling')
    return {'schema': 'village-001e-global-provider-allocation-accounting-v1',
        'manifest_sha256': allocation['original_manifest']['sha256'], 'allocation_manifest_sha256': sha256_file(path),
        'baseline_run_id': allocation['prior_recovery_run']['id'], 'baseline_artifact_sha256': allocation['prior_recovery_artifact']['digest'],
        'budget': allocation['budget'], 'actual_logical_requests': total,
        'actual_transport_launches': sum(row['transport_launches'] for row in results),
        'actual_supplemental_logical_requests': sum(row['supplemental_logical_requests'] for row in results),
        'actual_post_allocation_logical_requests': total - allocation['budget']['baseline_logical_requests_archived'],
        'all_48_lanes_reconstructed_offline': True, 'original_identities_and_judgments_preserved': True,
        'original_recovery_identities_and_baseline_files_preserved': True,
        'token_budget_amendment': frozen.TOKEN_POLICY, 'sibling_output_format_amendment': frozen.FORMAT_POLICY,
        'scientific_gates_unchanged': True, 'lanes': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('manifest')
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    for name in ('prior-root', 'prior-zip', 'run-record', 'jobs-record', 'artifact-inventory', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--allocation-manifest', type=Path, required=True)
    p.add_argument('--kind', choices=('ecology', 'tournament'), required=True)
    p.add_argument('--lane', required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--offline', action='store_true')
    p = sub.add_parser('verify-complete')
    p.add_argument('--allocation-manifest', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'manifest':
        allocation = make_manifest(args)
        write_once(args.out, allocation)
        print(json.dumps(allocation['budget'], indent=2))
    else:
        allocation, original, config = load_allocation(args.allocation_manifest)
        if args.command == 'run':
            run_lane(args.allocation_manifest, allocation, original, config, lane_record(allocation, args.kind, args.lane), args.resume, args.out, offline=args.offline)
        else:
            result = verify_complete(args.allocation_manifest, allocation, original, config, args.artifacts)
            write_once(args.out, result)
            print(json.dumps({key: result[key] for key in ('actual_logical_requests', 'actual_post_allocation_logical_requests', 'all_48_lanes_reconstructed_offline')}, indent=2))


if __name__ == '__main__':
    main()
