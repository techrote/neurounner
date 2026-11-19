"""Offline launch, native-input custody and adapter publication regressions."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import village_001f_operator as op
import village_001f_judge as judge
HISTORY_GUARD = op.verify_execution_history


class OperatorFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.root = self.home / "repo"
        self.work = self.root / op.REL
        self.work.mkdir(parents=True)
        self.accepted, self.head, self.main = "a" * 40, "b" * 40, "c" * 40
        self.summary = b'{"schema":"accepted-e-fixture"}\n'
        (self.root / op.E_SUMMARY).parent.mkdir(parents=True)
        (self.root / op.E_SUMMARY).write_bytes(self.summary)
        names = ("worker-config.json", "worker-eval.json", "sprocket-config.json", "judge-config.json")
        for name in names:
            op.dump(self.work / name, {"contract": name})
        self.contracts = {name: op.sha(self.work / name) for name in names}
        self.req = {"schema": "village-001f-post-e-request-v1", "publish_branch": op.BRANCH,
            "phase": "generate", "execution_revision": 1, "input_release_tag": op.INPUT_TAG,
            "final_release_tag": op.FINAL_TAG, "maximum_training_lanes": 12,
            "maximum_optimizer_steps": 1440, "new_worker_generations": 300,
            "new_sprocket_generations": 36, "maximum_provider_logical_requests": 114,
            "resume_run_id": None, "stage_e_main_commit": self.accepted,
            "stage_e_summary_sha256": hashlib.sha256(self.summary).hexdigest(),
            "contract_sha256": self.contracts}
        self.save_request()
        self.output = self.home / "github-output.txt"
        self.cfg = {"members": op.MEMBERS, "arms": op.ARMS, "training": {"paired_execution": {
            "member_jobs": 6, "fresh_cli_processes_per_job": 2,
            "member_arm_order": {member: list(op.ARMS if index % 2 == 0 else reversed(op.ARMS)) for index, member in enumerate(op.MEMBERS)}}}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(op, "ROOT", self.root))
        self.stack.enter_context(mock.patch.object(op, "WORK", self.work))
        self.stack.enter_context(mock.patch.object(op, "load_contract", return_value=(self.cfg, {}, {}, {})))
        self.stack.enter_context(mock.patch.dict(os.environ, {
            "GITHUB_REPOSITORY": op.REPO, "GITHUB_REF_NAME": op.BRANCH,
            "GITHUB_EVENT_NAME": "push", "GITHUB_SHA": self.head,
            "GITHUB_RUN_ID": "101", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_OUTPUT": str(self.output), "GH_TOKEN": "fixture-token"}))
        self.git_output = self.stack.enter_context(mock.patch.object(op.subprocess, "check_output", side_effect=self.remote_read))
        self.git_run = self.stack.enter_context(mock.patch.object(op.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)))
        self.github = self.stack.enter_context(mock.patch.object(op, "gh", return_value={"state": "closed", "state_reason": "completed"}))
        self.history = self.stack.enter_context(mock.patch.object(op, "verify_execution_history"))
        self.absent = self.stack.enter_context(mock.patch.object(op, "absent_tag"))
        self.download = self.stack.enter_context(mock.patch.object(op, "get_bytes", side_effect=AssertionError("live HTTP forbidden")))

    def save_request(self):
        op.dump(self.work / "request.json", self.req)

    def remote_read(self, command, **kwargs):
        if command == ["git", "ls-remote", "origin", "refs/heads/" + op.BRANCH]:
            return self.head + "\trefs/heads/" + op.BRANCH + "\n"
        if command == ["git", "ls-remote", "origin", "refs/heads/main"]:
            return self.main + "\trefs/heads/main\n"
        if command == ["git", "show", self.accepted + ":" + op.E_SUMMARY]:
            return self.summary
        raise AssertionError("unexpected external read: " + repr(command))

    def native_inputs(self):
        destination = self.work / "inputs"
        op.dump(destination / "worker/lane/rows/001-Q01.json", {"generated_token_ids": [4, 2]})
        record = {"schema": "village-001f-input-custody-v1", "run_id": 101,
            "source_head": self.head, "request_sha256": op.sha(self.work / "request.json"),
            "worker_generations": 300, "sprocket_generations": 36, "optimizer_steps": 1440,
            "files": op.files(destination)}
        record.update({key: self.req[key] for key in ("stage_e_main_commit", "stage_e_summary_sha256",
            "contract_sha256", "input_release_tag", "final_release_tag")})
        op.dump(self.work / "input-custody.json", record)
        return record


class LaunchGuardTests(OperatorFixture):
    def test_fixed_contracts_budgets_and_release_tags_are_required(self):
        self.assertEqual(op.request(), self.req)
        mutations = (
            {"maximum_provider_logical_requests": 115}, {"maximum_training_lanes": 13},
            {"maximum_optimizer_steps": 1441}, {"new_worker_generations": 301},
            {"input_release_tag": op.FINAL_TAG}, {"final_release_tag": op.INPUT_TAG},
            {"resume_run_id": True}, {"execution_revision": True}, {"contract_sha256": {}},
        )
        for change in mutations:
            with self.subTest(change=change):
                op.dump(self.work / "request.json", {**self.req, **change})
                with self.assertRaises(RuntimeError):
                    op.request()
        self.save_request()
        (self.work / "worker-eval.json").write_text("changed reserve\n")
        with self.assertRaisesRegex(RuntimeError, "contract changed"):
            op.request()

    def test_wrong_execution_surface_never_reads_live_authorities(self):
        for key, bad in (("GITHUB_REPOSITORY", "other/repo"), ("GITHUB_REF_NAME", "main"),
                         ("GITHUB_EVENT_NAME", "pull_request")):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: bad}):
                with self.assertRaisesRegex(RuntimeError, "execution surface"):
                    op.verify()
        self.git_output.assert_not_called()
        self.github.assert_not_called()

    def test_owned_branch_lease_precedes_issue_read(self):
        self.git_output.return_value = "d" * 40 + "\trefs/heads/" + op.BRANCH + "\n"
        self.git_output.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "branch moved"):
            op.verify()
        self.github.assert_not_called()
        self.git_run.assert_not_called()

    def test_closed_completed_stage_e_is_required(self):
        for state, reason in (("open", None), ("closed", "not_planned"), ("closed", None)):
            self.github.return_value = {"state": state, "state_reason": reason}
            with self.subTest(state=state, reason=reason), self.assertRaisesRegex(RuntimeError, "Stage E must be accepted"):
                op.verify()
        self.git_run.assert_not_called()
        self.absent.assert_not_called()

    def test_accepted_e_commit_must_be_on_current_main_and_execution_head(self):
        for failing_tip in (self.main, "HEAD"):
            def reject_wrong_ancestry(command, **kwargs):
                if command == ["git", "merge-base", "--is-ancestor", self.accepted, failing_tip]:
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0)
            self.git_run.side_effect = reject_wrong_ancestry
            with self.subTest(tip=failing_tip), self.assertRaises(subprocess.CalledProcessError):
                op.verify()
        self.assertFalse(self.output.exists())
        self.absent.assert_not_called()

    def test_accepted_commit_summary_must_match_declared_bytes(self):
        def changed_summary(command, **kwargs):
            return b"different accepted evidence" if command[1] == "show" else self.remote_read(command, **kwargs)
        self.git_output.side_effect = changed_summary
        with self.assertRaisesRegex(RuntimeError, "accepted main commit"):
            op.verify()
        self.assertFalse(self.output.exists())

    def test_valid_generate_guard_checks_both_ancestries_and_two_isolated_tags(self):
        op.verify()
        self.git_run.assert_any_call(["git", "merge-base", "--is-ancestor", self.accepted, self.main], check=True)
        self.git_run.assert_any_call(["git", "merge-base", "--is-ancestor", self.accepted, "HEAD"], check=True)
        self.assertEqual(self.absent.call_args_list, [mock.call(op.INPUT_TAG), mock.call(op.FINAL_TAG)])
        outputs = dict(line.split("=", 1) for line in self.output.read_text().splitlines())
        self.assertEqual(outputs["phase"], "generate")
        self.assertEqual(json.loads(outputs["members"]), op.MEMBERS)
        self.assertEqual(json.loads(outputs["arms"]), op.ARMS)
        self.assertEqual(json.loads(outputs["train_pairs"]), op.training_pairs(self.cfg))

    def test_judge_phase_requires_frozen_inputs_and_verified_input_release(self):
        self.native_inputs()
        self.req.update(phase="judge", execution_revision=2)
        self.save_request()
        with mock.patch.object(op, "verify_release") as verify_release:
            op.verify()
        verify_release.assert_called_once_with(self.work / "input-release-assets.json", op.INPUT_TAG)
        self.git_run.assert_any_call(["git", "merge-base", "--is-ancestor", self.head, "HEAD"], check=True)
        self.assertEqual(self.absent.call_args_list, [mock.call(op.FINAL_TAG)])


class InputCustodyTests(OperatorFixture):
    def test_phase_change_preserves_frozen_inputs_but_byte_change_is_refused(self):
        expected = self.native_inputs()
        self.req.update(phase="judge", execution_revision=2)
        self.save_request()
        self.assertEqual(op.verify_inputs(), expected)
        (self.work / "inputs/worker/lane/rows/001-Q01.json").write_text("changed native tokens")
        with self.assertRaisesRegex(RuntimeError, "custody changed"):
            op.verify_inputs()

    def test_frozen_authorities_cannot_be_redeclared_between_phases(self):
        record = self.native_inputs()
        for field, value in (("stage_e_main_commit", "d" * 40), ("contract_sha256", {}),
                             ("input_release_tag", op.FINAL_TAG), ("source_head", "not-a-head")):
            with self.subTest(field=field):
                op.dump(self.work / "input-custody.json", {**record, field: value})
                with self.assertRaises(RuntimeError):
                    op.verify_inputs()

    def test_native_evidence_rejects_weight_files_and_symlinks(self):
        for relative in ("worker/adapter_model.safetensors", "worker/optimizer.PT", "worker/adapter/README.md",
                         "worker/packed-adapter.tar.gz", "worker/native-or-weights.zip", "worker/Models/README.md"):
            root = self.home / hashlib.sha256(relative.encode()).hexdigest()[:8]
            (root / relative).parent.mkdir(parents=True)
            (root / relative).write_bytes(b"fixture")
            with self.subTest(relative=relative), self.assertRaisesRegex(RuntimeError, "outside Git"):
                op.evidence_files(root)
        links = self.home / "links"
        links.mkdir()
        (links / "receipt.json").symlink_to(self.work / "request.json")
        with self.assertRaisesRegex(RuntimeError, "nonregular"):
            op.evidence_files(links)

    def preflight_fixture(self):
        artifacts = self.home / "preflight"
        for name in ("worker", "sprocket", "training", "token-audit"):
            (artifacts / name).mkdir(parents=True)
        pairs = [{"pair_id": f"pair-{i:02d}", "scenario_id": f"T{i:02d}"} for i in range(60)]
        def identity(_path, _cfg, *, kind, member=None, stage=None, adapter_sha=None):
            return {"source": "fixed-fixture", "kind": kind, "member_id": member,
                    "stage": stage, "adapter_directory_sha256": adapter_sha}
        records = []
        for member in op.MEMBERS:
            for row in pairs:
                for choice in ("chosen", "rejected"):
                    ids = [1, 123, 2, 9, 2] + ([] if choice == "chosen" else [7, 2])
                    records.append({"member_id": member, "pair_id": row["pair_id"], "scenario_id": row["scenario_id"],
                        "choice": choice, "input_token_ids": ids, "sequence_tokens": len(ids),
                        "prompt_token_ids": ids[:3], "prompt_common_prefix_tokens": 3, "labels": [-100] * 3 + ids[3:],
                        "historical_attention_mask": [int(x != 2) for x in ids], "length_attention_mask": [1] * len(ids),
                        "real_tokens_masked_by_historical_rule": [i for i, x in enumerate(ids) if x == 2]})
        audit = {"identity": identity(None, None, kind="tokenizer-mask-audit"), "optimizer_steps": 0,
            "new_model_generations": 0, "tokenizer_eos_id": 2, "tokenizer_pad_id": 2,
            "sequences": 720, "records": records}
        path = artifacts / "token-audit/token-mask-audit.json"
        op.dump(path, audit)
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(mock.patch.object(judge, "load_contract", return_value=({}, self.cfg, {}, {}, {})))
        training = {i: {} for i in range(12)}
        workers = {i: {"rows": [{} for _ in range(12)]} for i in range(25)}
        train_mock = patches.enter_context(mock.patch.object(judge, "load_training", return_value=(training, {})))
        worker_mock = patches.enter_context(mock.patch.object(judge, "load_workers", return_value=(workers, {})))
        sprocket_mock = patches.enter_context(mock.patch.object(judge, "load_sprocket", return_value=([{}] * 36, {}, {})))
        patches.enter_context(mock.patch.object(op.worker, "run_identity", side_effect=identity))
        patches.enter_context(mock.patch.object(op.worker, "jsonl", return_value=pairs))
        patches.enter_context(mock.patch.object(op.worker, "selected_pairs", return_value=pairs[:8]))
        patches.enter_context(mock.patch.object(op.worker, "source_result", return_value={"adapter": {"sha256": "d" * 64}}))
        patches.enter_context(mock.patch.object(op.worker, "sha", return_value="a" * 64))
        keyed = {(row["member_id"], row["pair_id"], row["choice"]): row for row in records}
        for member in op.MEMBERS:
            for stage in ("BASE", "C", "D_ORIGINAL"):
                rows = []
                for pair in pairs[:8]:
                    chosen, rejected = [keyed[(member, pair["pair_id"], choice)] for choice in ("chosen", "rejected")]
                    ids = [chosen["input_token_ids"] + [2, 2], rejected["input_token_ids"]]
                    rows.append({"pair_id": pair["pair_id"], "scenario_id": pair["scenario_id"], "training_diagnostic": True,
                        "input_ids": ids, "labels": [chosen["labels"] + [-100, -100], rejected["labels"]],
                        "historical_attention_mask": [[int(x != 2) for x in seq] for seq in ids],
                        "length_attention_mask": [[1] * 5 + [0, 0], [1] * 7],
                        "real_positions_wrongly_masked": [[2, 4], [2, 4, 6]],
                        "scores": {arm: {"chosen_mean_logp": -1.0, "rejected_mean_logp": -2.0, "margin": 1.0} for arm in op.ARMS}})
                doc = {"identity": identity(None, None, kind="forward-only-paired-mask-scores", member=member, stage=stage,
                        adapter_sha=None if stage == "BASE" else "d" * 64), "pair_dataset_sha256": "a" * 64,
                       "model_forward_batches": 16, "optimizer_steps": 0, "new_generations": 0, "rows": rows}
                op.dump(artifacts / "worker" / member / "scores" / stage / "score.json", doc)
        return artifacts, path, audit, (train_mock, worker_mock, sprocket_mock)

    def test_native_preflight_requires_all_three_shared_source_validators(self):
        artifacts, _, _, validators = self.preflight_fixture()
        op.validate_input_sources(artifacts)
        for validate in validators:
            validate.assert_called_once()
            validate.side_effect = ValueError("source binding rejected")
            with self.assertRaisesRegex(ValueError, "source binding"):
                op.validate_input_sources(artifacts)
            validate.side_effect = None

    def test_native_token_audit_rejects_false_mask_positions_and_missing_pairs(self):
        artifacts, path, original, _ = self.preflight_fixture()
        mutations = (
            lambda a: a["records"][0].update(real_tokens_masked_by_historical_rule=[2]),
            lambda a: a["records"][0].update(length_attention_mask=[1, 1, 0, 1, 0]),
            lambda a: a["records"][0].update(scenario_id="wrong-case"),
            lambda a: a["records"].__setitem__(0, copy.deepcopy(a["records"][1])),
            lambda a: a["records"].pop(),
        )
        for mutation in mutations:
            audit = copy.deepcopy(original)
            mutation(audit)
            op.dump(path, audit)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                op.validate_input_sources(artifacts)

    def test_forward_scores_bind_native_ids_padding_masks_and_fixed_pair_selection(self):
        artifacts, _, _, _ = self.preflight_fixture()
        op.validate_input_sources(artifacts)
        path = artifacts / "worker" / op.MEMBERS[0] / "scores/C/score.json"
        original = json.loads(path.read_text())
        mutations = (
            lambda d: d["identity"].update(adapter_directory_sha256="b" * 64),
            lambda d: d["rows"][0]["input_ids"][0].__setitem__(2, 8),
            lambda d: d["rows"][0]["length_attention_mask"][0].__setitem__(2, 0),
            lambda d: d["rows"][0].update(pair_id="different-pair"),
            lambda d: d["rows"][0]["scores"][op.ARMS[0]].update(chosen_mean_logp=float("nan")),
            lambda d: d["rows"][0]["scores"][op.ARMS[0]].update(margin=2.0),
        )
        for mutate in mutations:
            doc = copy.deepcopy(original)
            mutate(doc)
            op.dump(path, doc)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(RuntimeError, "fixed-weight"):
                op.validate_input_sources(artifacts)
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "18 fixed-weight"):
            op.validate_input_sources(artifacts)


class ExecutionHistoryTests(OperatorFixture):
    def prior_spending(self, *, phase="generate", run_id=91, head="d" * 40):
        run = {"id": run_id, "path": op.WORKFLOW, "head_branch": op.BRANCH, "event": "push",
               "head_sha": head, "status": "completed"}
        job = {"name": "train (V-G01-A, historical_token_id_mask)" if phase == "generate" else "judge",
               "started_at": "2026-11-19T12:00:00Z", "status": "completed", "conclusion": "failure",
               "steps": [{"name": "Train one fixed 120-step matched arm" if phase == "generate" else "Blindly measure causal effects and feedback grounding",
                          "status": "completed", "conclusion": "failure"}]}
        artifact = {"id": 700 + run_id, "name": "village001f-train-V-G01-A-historical_token_id_mask" if phase == "generate" else "village001f-judgments",
            "expired": False, "workflow_run": {"id": run_id, "head_sha": head}, "size_in_bytes": 10,
            "digest": "sha256:" + "f" * 64,
            "archive_download_url": f"https://api.github.com/repos/{op.REPO}/actions/artifacts/{700 + run_id}/zip"}
        return run, job, artifact, {**self.req, "phase": phase}

    def history_fixture(self, cases):
        by_id = {case[0]["id"]: case for case in cases}
        def github(resource):
            if resource.startswith("actions/workflows/"):
                return {"total_count": len(cases), "workflow_runs": [case[0] for case in cases]}
            parts = resource.split("/")
            case = by_id[int(parts[2])]
            if parts[3].startswith("jobs?"):
                return {"total_count": 1, "jobs": [case[1]]}
            if parts[3].startswith("artifacts?"):
                artifacts = case[2] if isinstance(case[2], list) else [case[2]]
                return {"total_count": len(artifacts), "artifacts": artifacts}
            raise AssertionError("unexpected history read")
        self.github.side_effect = github
        def source_request(command, **kwargs):
            for case in cases:
                if command == ["git", "show", case[0]["head_sha"] + ":" + (op.REL / "request.json").as_posix()]:
                    return json.dumps(case[3]).encode()
            return self.remote_read(command, **kwargs)
        self.git_output.side_effect = source_request

    def test_actions_rerun_is_refused_before_history_reads(self):
        with mock.patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            with self.assertRaisesRegex(RuntimeError, "reruns are disabled"):
                HISTORY_GUARD(self.req)
        self.github.assert_not_called()

    def test_generate_and_judge_have_separate_initial_budgets(self):
        for current, prior in (("judge", "generate"), ("generate", "judge")):
            with self.subTest(current=current):
                self.history_fixture([self.prior_spending(phase=prior)])
                HISTORY_GUARD({**self.req, "phase": current})
        self.git_output.assert_not_called()

    def test_second_execution_must_resume_latest_spending_run(self):
        cases = [self.prior_spending(run_id=90, head="e" * 40), self.prior_spending(run_id=91)]
        self.history_fixture(cases)
        for source in (None, 90):
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "latest prior spending"):
                HISTORY_GUARD({**self.req, "resume_run_id": source})
        HISTORY_GUARD({**self.req, "resume_run_id": 91})
        self.git_run.assert_any_call(["git", "merge-base", "--is-ancestor", "d" * 40, "HEAD"], check=True)

    def test_recovery_history_requires_same_contracts_and_unexpired_artifacts(self):
        case = self.prior_spending()
        self.history_fixture([case])
        request = {**self.req, "resume_run_id": 91}
        case[3]["contract_sha256"] = {}
        with self.assertRaisesRegex(RuntimeError, "different frozen authority"):
            HISTORY_GUARD(request)
        case[3]["contract_sha256"] = self.contracts
        case[2]["expired"] = True
        with self.assertRaisesRegex(RuntimeError, "expiry mismatch"):
            HISTORY_GUARD(request)

    def test_latest_resume_cannot_drop_an_earlier_spending_lane(self):
        older = self.prior_spending(run_id=90, head="e" * 40)
        older[1]["name"] = "base"
        older[1]["steps"][0]["name"] = "Generate fixed Q BASE control"
        older[2]["name"] = "village001f-infer-BASE"
        self.history_fixture([older, self.prior_spending(run_id=91)])
        with self.assertRaisesRegex(RuntimeError, "lost a previously spending lane"):
            HISTORY_GUARD({**self.req, "resume_run_id": 91})

    def test_refused_actions_rerun_cannot_hide_spending_in_the_first_attempt(self):
        case = self.prior_spending()
        case[0]["run_attempt"] = 2
        self.history_fixture([case])
        original_read = self.github.side_effect
        def all_attempts(resource):
            if "/jobs?" in resource:
                self.assertIn("filter=all", resource)
                skipped = {**case[1], "conclusion": "skipped", "steps": []}
                return {"total_count": 2, "jobs": [skipped, case[1]]}
            return original_read(resource)
        self.github.side_effect = all_attempts
        with self.assertRaisesRegex(RuntimeError, "latest prior spending"):
            HISTORY_GUARD(self.req)
        HISTORY_GUARD({**self.req, "resume_run_id": 91})

    def test_incomplete_history_and_missing_execution_steps_fail_closed(self):
        self.github.return_value = {"total_count": 101, "workflow_runs": []}
        with self.assertRaisesRegex(RuntimeError, "bounded diagnostic run history"):
            HISTORY_GUARD(self.req)
        case = self.prior_spending()
        case[1]["steps"] = []
        self.history_fixture([case])
        with self.assertRaisesRegex(RuntimeError, "lacks its bounded execution step"):
            HISTORY_GUARD(self.req)

    def paired_spending(self, member, *, second=False):
        run, job, artifact, request = self.prior_spending()
        order = op.worker.training_arm_order(self.cfg, member)
        job["name"] = f"train ({member})"
        job["steps"] = [{"name": op.PAIRED_TRAIN_STEPS[0], "status": "completed", "conclusion": "success" if second else "failure"},
                        {"name": op.PAIRED_TRAIN_STEPS[1], "status": "completed", "conclusion": "failure" if second else "skipped"}]
        artifacts = []
        for index, arm in enumerate(order[:2 if second else 1]):
            item = {**artifact, "id": 900 + index, "name": f"village001f-train-{member}-{arm}",
                    "archive_download_url": f"https://api.github.com/repos/{op.REPO}/actions/artifacts/{900 + index}/zip"}
            artifacts.append(item)
        return run, job, artifacts, request

    def test_counterbalanced_first_arm_failure_does_not_spend_skipped_second(self):
        self.assertEqual([row["first_arm"] for row in op.training_pairs(self.cfg)], [op.ARMS[i % 2] for i in range(6)])
        for member in op.MEMBERS:
            with self.subTest(member=member):
                case = self.paired_spending(member)
                self.history_fixture([case])
                HISTORY_GUARD({**self.req, "resume_run_id": 91})
                case[2][0]["name"] = f"village001f-train-{member}-{op.worker.training_arm_order(self.cfg, member)[1]}"
                with self.assertRaisesRegex(RuntimeError, "lost a previously spending lane"):
                    HISTORY_GUARD({**self.req, "resume_run_id": 91})

    def test_second_entered_arm_requires_both_artifacts_and_successful_first(self):
        case = self.paired_spending(op.MEMBERS[1], second=True)
        self.history_fixture([case])
        HISTORY_GUARD({**self.req, "resume_run_id": 91})
        second = case[2].pop()
        with self.assertRaisesRegex(RuntimeError, "lost a previously spending lane"):
            HISTORY_GUARD({**self.req, "resume_run_id": 91})
        case[2].append(second)
        case[1]["steps"][0]["conclusion"] = "failure"
        with self.assertRaisesRegex(RuntimeError, "without a successful first"):
            HISTORY_GUARD({**self.req, "resume_run_id": 91})

    def test_paired_history_requires_explicit_second_step_visibility(self):
        case = self.paired_spending(op.MEMBERS[0])
        case[1]["steps"].pop()
        self.history_fixture([case])
        with self.assertRaisesRegex(RuntimeError, "show both arm steps"):
            HISTORY_GUARD({**self.req, "resume_run_id": 91})

    def test_older_spending_order_contract_cannot_be_remapped(self):
        older = self.prior_spending(run_id=90, head="e" * 40)
        older[3]["contract_sha256"] = {**self.contracts, "worker-config.json": "f" * 64}
        self.history_fixture([older, self.prior_spending(run_id=91)])
        with self.assertRaisesRegex(RuntimeError, "different frozen authority"):
            HISTORY_GUARD({**self.req, "resume_run_id": 91})


class PairedStagingTests(OperatorFixture):
    def test_prior_second_arm_survives_first_arm_failure_with_latest_partial(self):
        resume, work, out = [self.home / name for name in ("resume", "work", "out")]
        for arm in op.ARMS:
            op.dump(resume / arm / "lane/checkpoint.json", {"mask_mode": arm, "completed_steps": 60})
            op.dump(resume / arm / "lane/training-invocations/invocation-0001-start.json", {"prior": arm})
        original_second = op.files(resume / op.ARMS[1] / "lane")
        op.stage_training(op.MEMBERS[0], resume, work, out)
        self.assertEqual(op.files(out / op.ARMS[1]), original_second)
        op.dump(work / op.ARMS[0] / "checkpoint.json", {"mask_mode": op.ARMS[0], "completed_steps": 90})
        op.dump(work / op.ARMS[0] / "training-invocations/invocation-0002-start.json", {"interrupted": True})
        op.stage_training(op.MEMBERS[0], resume, work, out)
        self.assertEqual(op.files(out / op.ARMS[0]), op.files(work / op.ARMS[0]))
        self.assertEqual(op.files(out / op.ARMS[1]), original_second)
        self.assertEqual(json.loads((resume / op.ARMS[0] / "lane/checkpoint.json").read_text())["completed_steps"], 60)

    def test_unentered_fresh_second_arm_is_not_fabricated(self):
        resume, work, out = [self.home / name for name in ("resume", "work", "out")]
        op.stage_training(op.MEMBERS[1], resume, work, out)
        self.assertFalse(out.exists())
        first, second = op.worker.training_arm_order(self.cfg, op.MEMBERS[1])
        op.dump(work / first / "training-identity.json", {"mask_mode": first})
        op.stage_training(op.MEMBERS[1], resume, work, out)
        self.assertTrue((out / first / "training-identity.json").is_file())
        self.assertFalse((out / second).exists())

    def test_paired_staging_rejects_overlapping_roots_and_symlinks(self):
        with self.assertRaisesRegex(RuntimeError, "must be separate"):
            op.stage_training(op.MEMBERS[0], self.home / "resume", self.home / "work", self.home / "work/out")
        (self.home / "work").mkdir()
        (self.home / "work" / op.ARMS[0]).symlink_to(self.work)
        with self.assertRaisesRegex(RuntimeError, "nonregular"):
            op.stage_training(op.MEMBERS[0], self.home / "resume", self.home / "work", self.home / "out")


class ArchivePublicationTests(OperatorFixture):
    def archive_fixture(self):
        artifacts = self.home / "artifacts"
        for member in op.MEMBERS:
            for stage in ("C", "D_ORIGINAL", "D_REPRODUCED", "D_CORRECTED"):
                op.dump(artifacts / "worker" / f"village001f-infer-{member}" / stage / "result.json",
                        {"rows": [{"scenario_id": f"Q{i:02d}"} for i in range(1, 13)]})
            for arm in op.ARMS:
                lane = artifacts / "training" / f"village001f-train-{member}-{arm}"
                for name in ("training.json", "training-identity.json", "result.json"):
                    op.dump(lane / name, {"member_id": member, "mask_mode": arm, "file": name})
                for name in ("invocation-0001-start.json", "invocation-0001-checkpoint-000120.json", "invocation-0001-finish.json"):
                    op.dump(lane / "training-invocations" / name, {"member_id": member, "mask_mode": arm, "file": name})
                op.dump(lane / "checkpoint.json", {"transient": "do not publish"})
                op.dump(lane / "adapter/adapter_config.json", {"fixture": True})
                (lane / "adapter/adapter_model.safetensors").write_bytes((member + arm).encode())
        op.dump(artifacts / "worker/village001f-infer-BASE/result.json", {"rows": [{}] * 12})
        op.dump(artifacts / "sprocket/summary.json", {"local_generations": 36})
        (artifacts / "sprocket/native-response.txt").write_bytes(b"  exact raw native response\n\n")
        op.dump(artifacts / "token-audit/token-mask-audit.json", {"sequences": 720})
        def provenance(root, _cfg, *, require_complete=True):
            self.assertTrue(require_complete)
            return {}, {path.relative_to(root).as_posix(): path for path in sorted((root / "training-invocations").glob("*.json"))}
        self.stack.enter_context(mock.patch.object(op.worker, "validate_training_provenance", side_effect=provenance))
        return artifacts

    def test_source_validation_failure_precedes_all_publication_writes(self):
        artifacts = self.archive_fixture()
        release = self.home / "release"
        with mock.patch.object(op, "validate_input_sources", side_effect=ValueError("invalid native source")), \
             mock.patch.object(op, "checked_adapter") as adapter:
            with self.assertRaisesRegex(ValueError, "native source"):
                op.archive_inputs(artifacts, release)
        adapter.assert_not_called()
        self.assertFalse(release.exists())
        self.assertFalse((self.work / "inputs").exists())
        self.assertFalse((self.work / "input-custody.json").exists())

    def test_adapter_release_directory_must_be_outside_git(self):
        with mock.patch.object(op, "validate_input_sources") as validate:
            with self.assertRaisesRegex(RuntimeError, "outside repository"):
                op.archive_inputs(self.home / "unused", self.root / "adapters-release")
        validate.assert_not_called()

    def test_training_symlink_cannot_be_smuggled_into_adapter_archive(self):
        artifacts = self.archive_fixture()
        lane = artifacts / "training" / f"village001f-train-{op.MEMBERS[0]}-{op.ARMS[0]}"
        (lane / "adapter/linked-weights.bin").symlink_to(self.work / "request.json")
        with mock.patch.object(op, "validate_input_sources"), mock.patch.object(op, "checked_adapter"):
            with self.assertRaisesRegex(RuntimeError, "nonregular"):
                op.archive_inputs(artifacts, self.home / "release")
        self.assertFalse((self.home / "release").exists())
        self.assertFalse((self.work / "inputs").exists())

    def test_archives_preserve_twelve_adapters_outside_git_and_only_receipts_inside(self):
        artifacts = self.archive_fixture()
        release = self.home / "release"
        with mock.patch.object(op, "validate_input_sources") as validate, mock.patch.object(op, "checked_adapter") as adapter:
            op.archive_inputs(artifacts, release)
        validate.assert_called_once_with(artifacts)
        self.assertEqual(adapter.call_count, 12)
        record = op.verify_inputs()
        self.assertEqual(record["contract_sha256"], self.contracts)
        self.assertEqual(record["stage_e_main_commit"], self.accepted)
        self.assertEqual(len(list(release.glob("*.tar.gz"))), 12)
        self.assertFalse(any(Path(name).suffix in op.WEIGHT_SUFFIXES for name in record["files"]))
        for member in op.MEMBERS:
            for arm in op.ARMS:
                kept = self.work / "inputs/training" / member / arm
                self.assertEqual({p.name for p in kept.iterdir()}, {"training.json", "training-identity.json", "result.json", "training-invocations"})
                lane = artifacts / "training" / f"village001f-train-{member}-{arm}"
                self.assertEqual(op.files(kept / "training-invocations"), op.files(lane / "training-invocations"))
                with tarfile.open(release / f"village001f-{member}-{arm}.tar.gz") as archive:
                    self.assertEqual(set(archive.getnames()), {"adapter_config.json", "adapter_model.safetensors"})
                    for entry in archive.getmembers():
                        self.assertEqual((entry.uid, entry.gid, entry.mtime, entry.uname, entry.gname), (0, 0, 0, "", ""))
                        self.assertEqual(archive.extractfile(entry).read(), (lane / "adapter" / entry.name).read_bytes())
        self.assertEqual((self.work / "inputs/sprocket/native-response.txt").read_bytes(), b"  exact raw native response\n\n")
        with self.assertRaisesRegex(RuntimeError, "already frozen"):
            op.archive_inputs(artifacts, self.home / "second-release")


class ReleaseAndRecoveryTests(OperatorFixture):
    def test_release_verification_rejects_duplicates_drafts_wrong_tags_and_digest_drift(self):
        expected = {"release_tag": op.INPUT_TAG, "assets": [{"file": "adapter.tar.gz", "bytes": 7, "sha256": "a" * 64}]}
        manifest = self.work / "input-release-assets.json"
        op.dump(manifest, expected)
        valid = {"tag_name": op.INPUT_TAG, "prerelease": True, "draft": False,
            "assets": [{"name": "adapter.tar.gz", "size": 7, "digest": "sha256:" + "a" * 64, "state": "uploaded"}]}
        self.github.return_value = valid
        op.verify_release(manifest, op.INPUT_TAG)
        mutations = (lambda x: x.update(draft=True), lambda x: x.update(tag_name=op.FINAL_TAG),
            lambda x: x["assets"].append(copy.deepcopy(x["assets"][0])),
            lambda x: x["assets"][0].update(digest="sha256:" + "b" * 64))
        for mutate in mutations:
            case = copy.deepcopy(valid)
            mutate(case)
            self.github.return_value = case
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                op.verify_release(manifest, op.INPUT_TAG)
        with self.assertRaisesRegex(RuntimeError, "not isolated"):
            op.verify_release(manifest, "original-stage-d")

    def test_release_manifest_requires_regular_top_level_assets(self):
        directory = self.home / "release"
        directory.mkdir()
        (directory / "adapter.tar.gz").write_bytes(b"archive")
        manifest = self.work / "assets.json"
        op.release_inventory(directory, manifest, op.INPUT_TAG)
        self.assertEqual(json.loads(manifest.read_text())["assets"][0]["sha256"], op.sha(directory / "adapter.tar.gz"))
        op.dump(directory / "nested/receipt.json", {})
        with self.assertRaisesRegex(RuntimeError, "top-level"):
            op.release_inventory(directory, manifest, op.INPUT_TAG)

    def recovery_fixture(self, zip_members=None):
        self.req["resume_run_id"] = 42
        self.save_request()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in (zip_members or {"rows/001-Q01.json": b'{"native":[1,2]}\n'}).items():
                archive.writestr(name, payload)
        data = stream.getvalue()
        repo = {"id": 1432134981, "full_name": op.REPO}
        run = {"id": 42, "status": "completed", "path": op.WORKFLOW, "head_branch": op.BRANCH,
               "head_sha": self.head, "repository": repo, "head_repository": dict(repo)}
        url = f"https://api.github.com/repos/{op.REPO}/actions/artifacts/123/zip"
        artifact = {"id": 123, "name": "village001f-infer-BASE", "expired": False,
            "workflow_run": {"id": 42, "head_sha": self.head}, "size_in_bytes": len(data),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "archive_download_url": url}
        api = f"https://api.github.com/repos/{op.REPO}/actions/runs/42"
        def download(url_in, token, limit=None):
            self.assertEqual(token, "fixture-token")
            if url_in == api:
                return json.dumps(run).encode()
            if url_in == api + "/artifacts?per_page=100":
                return json.dumps({"total_count": 1, "artifacts": [artifact]}).encode()
            if url_in == url:
                self.assertEqual(limit, len(data))
                return data
            raise AssertionError("unexpected URL: " + url_in)
        self.download.side_effect = download
        return run, artifact, data

    def test_recovery_verifies_original_zip_and_retains_custody(self):
        _, artifact, data = self.recovery_fixture()
        out = self.home / "resume/lane"
        op.recover(artifact["name"], out)
        self.assertEqual(out.with_suffix(".zip").read_bytes(), data)
        self.assertEqual((out / "rows/001-Q01.json").read_bytes(), b'{"native":[1,2]}\n')
        custody = json.loads((out.parent / "lane-custody.json").read_text())
        self.assertEqual(custody["artifact"]["digest"], artifact["digest"])
        self.assertEqual(custody["files"][0]["sha256"], op.sha(out / "rows/001-Q01.json"))
        with self.assertRaisesRegex(RuntimeError, "ZIP already exists"):
            op.recover(artifact["name"], out)

    def test_recovery_rejects_wrong_source_and_expired_artifacts_before_download(self):
        run, artifact, _ = self.recovery_fixture()
        for target, key, bad in ((run, "head_branch", "main"), (run["head_repository"], "id", 9),
                                 (artifact, "expired", True), (artifact["workflow_run"], "head_sha", "d" * 40)):
            old = target[key]
            target[key] = bad
            self.download.reset_mock()
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                op.recover(artifact["name"], self.home / "resume/lane")
            self.assertLess(self.download.call_count, 3)
            self.assertFalse((self.home / "resume/lane").exists())
            target[key] = old

    def test_recovery_digest_mismatch_never_extracts_or_certifies(self):
        _, artifact, _ = self.recovery_fixture()
        artifact["digest"] = "sha256:" + "0" * 64
        out = self.home / "resume/lane"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            op.recover(artifact["name"], out)
        self.assertFalse(out.exists())
        self.assertFalse((out.parent / "lane-custody.json").exists())

    def test_recovery_path_traversal_never_extracts_or_certifies(self):
        _, artifact, _ = self.recovery_fixture({"../escape": b"bad", "rows/valid.json": b"{}"})
        out = self.home / "resume/lane"
        with self.assertRaisesRegex(ValueError, "traversal"):
            op.recover(artifact["name"], out)
        self.assertFalse(out.exists())
        self.assertFalse((out.parent / "escape").exists())
        self.assertFalse((out.parent / "lane-custody.json").exists())

    def test_permanent_recovery_keeps_original_zip_outside_git_and_ignores_extracted_lane(self):
        _, artifact, original = self.recovery_fixture()
        out = self.home / "resume/lane"
        op.recover(artifact["name"], out)
        (out / "adapter_model.safetensors").write_bytes(b"transient extracted weights")
        release = self.home / "release"
        release.mkdir()
        evidence = self.work / "recovery/generate"
        op.archive_recovery(out.parent, evidence, release)
        manifest = json.loads((evidence / "recovery-manifest.json").read_text())
        self.assertEqual(len(manifest["sources"]), 1)
        source = manifest["sources"][0]
        self.assertEqual((release / source["archive_file"]).read_bytes(), original)
        self.assertEqual((evidence / source["receipt_file"]).read_bytes(), (out.parent / "lane-custody.json").read_bytes())
        self.assertFalse(list(evidence.rglob("*.zip")))
        self.assertFalse(list(evidence.rglob("*.safetensors")))
        with self.assertRaisesRegex(RuntimeError, "must be new"):
            op.archive_recovery(out.parent, evidence, release)

    def test_corrupt_recovery_receipt_or_orphan_zip_cannot_enter_permanent_evidence(self):
        _, artifact, _ = self.recovery_fixture()
        out = self.home / "resume/lane"
        op.recover(artifact["name"], out)
        receipt_path = out.parent / "lane-custody.json"
        original = receipt_path.read_bytes()
        receipt = json.loads(original)
        receipt["files"] = []
        op.dump(receipt_path, receipt)
        release = self.home / "release"
        release.mkdir()
        evidence = self.work / "recovery/generate"
        with self.assertRaisesRegex(RuntimeError, "extracted-file custody"):
            op.archive_recovery(out.parent, evidence, release)
        self.assertFalse(evidence.exists())
        self.assertEqual(list(release.iterdir()), [])
        receipt_path.write_bytes(original)
        (out.parent / "orphan.zip").write_bytes(b"unbound ZIP")
        with self.assertRaisesRegex(RuntimeError, "unbound files"):
            op.archive_recovery(out.parent, evidence, release)

    def test_second_recovery_preserves_every_prior_native_zip_and_receipt(self):
        run, artifact, data = self.recovery_fixture()
        older_run = {**run, "id": 40, "head_sha": "d" * 40}
        older_artifact = {**artifact, "id": 99, "workflow_run": {"id": 40, "head_sha": "d" * 40},
            "archive_download_url": f"https://api.github.com/repos/{op.REPO}/actions/artifacts/99/zip"}
        older_receipt = {"run": older_run, "artifact": older_artifact, "files": [{"path": "rows/001-Q01.json",
            "bytes": len(b'{"native":[1,2]}\n'), "sha256": hashlib.sha256(b'{"native":[1,2]}\n').hexdigest()}]}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("lane.zip", data)
            archive.writestr("lane-custody.json", json.dumps(older_receipt).encode())
        inherited_data = stream.getvalue()
        history_artifact = {**artifact, "id": 124, "name": "village001f-recovery-infer-BASE",
            "size_in_bytes": len(inherited_data), "digest": "sha256:" + hashlib.sha256(inherited_data).hexdigest(),
            "archive_download_url": f"https://api.github.com/repos/{op.REPO}/actions/artifacts/124/zip"}
        def download(url, token, limit=None):
            if url.endswith("/runs/42"):
                return json.dumps(run).encode()
            if url.endswith("/artifacts?per_page=100"):
                return json.dumps({"total_count": 2, "artifacts": [artifact, history_artifact]}).encode()
            if url.endswith("/123/zip"):
                return data
            if url.endswith("/124/zip"):
                return inherited_data
            raise AssertionError(url)
        self.download.side_effect = download
        out = self.home / "resume/lane"
        op.recover(artifact["name"], out)
        self.assertEqual((out.parent / "history/lane.zip").read_bytes(), data)
        release = self.home / "release"
        release.mkdir()
        evidence = self.work / "recovery/generate"
        op.archive_recovery(out.parent, evidence, release)
        manifest = json.loads((evidence / "recovery-manifest.json").read_text())
        self.assertEqual({source["artifact_id"] for source in manifest["sources"]}, {99, 123})
        self.assertEqual(len(list(release.glob("*.zip"))), 2)
        self.assertEqual(len(list(evidence.glob("*-custody.json"))), 2)


if __name__ == "__main__":
    unittest.main()
