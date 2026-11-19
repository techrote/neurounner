"""Offline custody tests; synthetic archives never contain model weights."""
from __future__ import annotations

import copy
import io
import json
import stat
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

from scripts import village_001d_recover_completed as recovery


def source_run():
    repo = {"id": recovery.REPOSITORY_ID, "full_name": recovery.REPOSITORY}
    return {"id": recovery.SOURCE_RUN, "head_sha": recovery.SOURCE_HEAD,
            "head_branch": "main", "path": recovery.WORKFLOW_PATH,
            "status": "completed", "conclusion": "failure", "event": "workflow_dispatch",
            "run_attempt": 1, "repository": repo, "head_repository": dict(repo),
            "head_commit": {"id": recovery.SOURCE_HEAD}}


def source_jobs():
    names = ["plan"] + [f"{stage} ({m})" for stage in
                        ("feedback", "compile_pairs", "preference_train", "compare")
                        for m in recovery.MEMBERS] + ["assemble"]
    jobs = []
    for index, name in enumerate(names):
        failed = name == f"compare ({recovery.FAILED_MEMBER})"
        jobs.append({"id": recovery.FAILED_JOB if failed else 1000 + index, "name": name,
                     "run_id": recovery.SOURCE_RUN, "head_sha": recovery.SOURCE_HEAD,
                     "status": "completed", "run_attempt": 1,
                     "conclusion": "failure" if failed else "skipped" if name == "assemble" else "success"})
    return jobs


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def artifact(name, data, aid):
    return {"id": aid, "name": name, "size_in_bytes": len(data),
            "expired": False, "expires_at": "2099-01-01T00:00:00Z",
            "digest": "sha256:" + recovery.sha256(data),
            "archive_download_url": f"https://api.github.com/repos/{recovery.REPOSITORY}/actions/artifacts/{aid}/zip",
            "workflow_run": {"id": recovery.SOURCE_RUN, "head_sha": recovery.SOURCE_HEAD,
                             "head_branch": "main", "repository_id": recovery.REPOSITORY_ID,
                             "head_repository_id": recovery.REPOSITORY_ID}}


def population(zips=None):
    rows = []
    for stage in recovery.STAGES:
        for member in recovery.MEMBERS:
            if stage == "feedback":
                files = {"manifest.json": b"{}\n", "feedback.jsonl": b"{}\n"}
            elif stage == "pairs":
                files = {"manifest.json": b"{}\n", "pairs.jsonl": b"{}\n"}
            elif stage == "preference":
                files = {"result.json": b"{}\n"}
            else:
                files = {"provider/raw.txt": b"  original raw response  \n\n"}
                if member != recovery.FAILED_MEMBER:
                    files["evaluation.json"] = b"{}\n"
            data = zip_bytes(files)
            name = f"village001d-{stage}-{member}"
            rows.append(artifact(name, data, 10000 + len(rows)))
            if zips is not None:
                (zips / f"{name}.zip").write_bytes(data)
    return rows


class SourceCustodyTests(unittest.TestCase):
    def test_exact_original_metadata_and_terminal_coverage(self):
        recovery.validate_source_run(source_run())
        recovery.validate_jobs(source_jobs())
        self.assertEqual(len(recovery.validate_inventory(population())), 144)

    def test_wrong_run_source_repo_workflow_or_attempt_refused(self):
        for key, value in (("id", 1), ("head_sha", "0" * 40), ("path", "other.yml"),
                           ("head_branch", "other"), ("conclusion", "success"), ("run_attempt", 2)):
            with self.subTest(key=key):
                run = source_run()
                run[key] = value
                with self.assertRaises(ValueError):
                    recovery.validate_source_run(run)
        run = source_run()
        run["head_repository"]["id"] = 7
        with self.assertRaises(ValueError):
            recovery.validate_source_run(run)

    def test_missing_failed_or_duplicate_completed_job_refused(self):
        for mutate in (lambda rows: rows.pop(),
                       lambda rows: rows[1].update(conclusion="failure"),
                       lambda rows: rows[1].update(head_sha="0" * 40),
                       lambda rows: rows[1].update(id=rows[0]["id"]),
                       lambda rows: rows[1].update(name=rows[0]["name"])):
            jobs = source_jobs()
            mutate(jobs)
            with self.assertRaises(ValueError):
                recovery.validate_jobs(jobs)

    def test_artifact_inventory_fails_closed(self):
        rows = population()
        for mutate in (lambda x: x.pop(),
                       lambda x: x.append(copy.deepcopy(x[0])),
                       lambda x: x[1].update(name=x[0]["name"]),
                       lambda x: x[1].update(id=x[0]["id"]),
                       lambda x: x[0].update(expired=True),
                       lambda x: x[0].update(expires_at="2000-01-01T00:00:00Z"),
                       lambda x: x[0].update(digest=None),
                       lambda x: x[0].update(size_in_bytes=0),
                       lambda x: x[0].update(archive_download_url="https://example.invalid/archive.zip"),
                       lambda x: x[0]["workflow_run"].update(head_sha="0" * 40)):
            case = copy.deepcopy(rows)
            mutate(case)
            with self.assertRaises((ValueError, TypeError)):
                recovery.validate_inventory(case)

    def test_full_population_recovery_preserves_every_original_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zips = root / "originals"
            zips.mkdir()
            rows = population(zips)
            with mock.patch.object(recovery, "get_bytes", side_effect=AssertionError("network forbidden")):
                result = recovery.recover(root / "recovered", source_run(), rows,
                                          source_jobs(), existing_zips=zips)
            self.assertEqual(result["original_artifacts"], 144)
            self.assertEqual(result["new_optimizer_steps"], 0)
            self.assertEqual(result["new_provider_calls"], 0)
            self.assertEqual(len(result["artifacts"]), 144)
            out = root / "recovered"
            self.assertEqual(len(list((out / "comparison").glob("*/evaluation.json"))), 35)
            self.assertEqual(len(list((out / "preference").glob("*/result.json"))), 36)
            for row in result["artifacts"]:
                original = zips / f"{row['name']}.zip"
                retained = out / "custody" / row["original_zip"]
                self.assertEqual(original.read_bytes(), retained.read_bytes())
                for member in row["files"]:
                    content = (out / row["working_copy"] / member["path"]).read_bytes()
                    self.assertEqual(recovery.sha256(content), member["sha256"])
                    self.assertEqual(len(content), member["bytes"])
            self.assertEqual((out / "comparison" / recovery.FAILED_MEMBER / "provider/raw.txt").read_bytes(),
                             b"  original raw response  \n\n")
            for name, record in result["metadata"].items():
                self.assertEqual(recovery.sha256((out / "custody" / name).read_bytes()), record["sha256"])
            with self.assertRaisesRegex(ValueError, "fresh"):
                recovery.recover(out, source_run(), rows, source_jobs(), existing_zips=zips)

    def test_corrupt_original_zip_never_gets_success_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zips = root / "originals"
            zips.mkdir()
            rows = population(zips)
            first = sorted(zips.iterdir())[0]
            content = bytearray(first.read_bytes())
            content[-1] ^= 1
            first.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                recovery.recover(root / "out", source_run(), rows, source_jobs(), existing_zips=zips)
            self.assertFalse((root / "out/custody/manifest.json").exists())

    def test_permanent_original_zips_remain_verifiable_after_actions_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zips = root / "originals"
            zips.mkdir()
            rows = population(zips)
            for row in rows:
                row.update(expires_at="2000-01-01T00:00:00Z", expired=True)
            with mock.patch.object(recovery, "get_bytes", side_effect=AssertionError("network forbidden")):
                result = recovery.recover(root / "offline", source_run(), rows,
                                          source_jobs(), existing_zips=zips)
                with self.assertRaisesRegex(ValueError, "expired"):
                    recovery.recover(root / "live", source_run(), rows, source_jobs())
            self.assertEqual(result["custody_mode"], "verified_existing_zips")
            self.assertEqual(result["original_artifacts"], 144)
            self.assertFalse((root / "live").exists())


class ZipBoundaryTests(unittest.TestCase):
    def test_transient_get_failure_retries_without_weakening_validation(self):
        failure = urllib.error.HTTPError("https://invalid.example", 502, "bad gateway", {}, None)
        with mock.patch.object(recovery, "_get_bytes_once", side_effect=[failure, b"original"]) as fetch, \
             mock.patch.object(recovery.time, "sleep") as sleep:
            self.assertEqual(recovery.get_bytes("source", "secret", 123), b"original")
        self.assertEqual(fetch.call_args_list, [mock.call("source", "secret", 123)] * 2)
        sleep.assert_called_once_with(1)

    def test_permanent_http_failure_is_not_retried(self):
        for status in (401, 403, 404, 410):
            with self.subTest(status=status):
                failure = urllib.error.HTTPError("https://invalid.example", status, "denied", {}, None)
                with mock.patch.object(recovery, "_get_bytes_once", side_effect=failure) as fetch, \
                     mock.patch.object(recovery.time, "sleep") as sleep:
                    with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                        recovery.get_bytes("source", "secret")
                self.assertEqual(fetch.call_count, 1)
                sleep.assert_not_called()

    def test_transient_get_retry_budget_is_four_attempts(self):
        failure = urllib.error.HTTPError("https://invalid.example", 503, "unavailable", {}, None)
        with mock.patch.object(recovery, "_get_bytes_once", side_effect=failure) as fetch, \
             mock.patch.object(recovery.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                recovery.get_bytes("source", "secret")
        self.assertEqual(fetch.call_count, 4)
        self.assertEqual(sleep.call_args_list, [mock.call(1), mock.call(2), mock.call(4)])

    def test_github_token_is_never_forwarded_to_artifact_storage(self):
        url = f"https://api.github.com/repos/{recovery.REPOSITORY}/actions/artifacts/1/zip"
        storage = "https://storage.example.invalid/archive.zip?sig=synthetic"
        redirect = urllib.error.HTTPError(url, 302, "Found", {"Location": storage}, None)
        opener = mock.Mock()
        opener.open.side_effect = redirect
        with mock.patch.object(recovery.urllib.request, "build_opener", return_value=opener), \
             mock.patch.object(recovery.urllib.request, "urlopen", return_value=io.BytesIO(b"zip")) as fetch:
            self.assertEqual(recovery.get_bytes(url, "synthetic-secret"), b"zip")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-secret")
        fetch.assert_called_once_with(storage, timeout=90)

    def test_path_traversal_symlink_duplicates_and_conflicts_refused_before_extract(self):
        scenarios = [
            [("../escape", b"bad")], [("/absolute", b"bad")], [("C:/absolute", b"bad")],
            [("foo\\bar", b"bad")], [("a/./b", b"bad")], [("a//b", b"bad")],
            [("same", b"one"), ("same", b"two")], [("parent", b"file"), ("parent/child", b"bad")],
        ]
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        scenarios.append([(symlink, b"../../outside")])
        for entries in scenarios:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    for name, content in entries:
                        stream.writestr(name, content)
                with self.assertRaises(ValueError):
                    recovery.safe_extract(archive, root / "extract")
                self.assertFalse((root / "extract").exists())
                self.assertFalse((root / "escape").exists())

    def test_original_zip_symlink_and_size_change_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = zip_bytes({"file.txt": b"bytes"})
            original = root / "original.zip"
            original.write_bytes(content)
            row = artifact("test", content, 1)
            link = root / "linked.zip"
            link.symlink_to(original)
            with self.assertRaises(ValueError):
                recovery.verify_zip(link, row)
            original.write_bytes(content + b"extra")
            with self.assertRaisesRegex(ValueError, "size"):
                recovery.verify_zip(original, row)


if __name__ == "__main__":
    unittest.main()
