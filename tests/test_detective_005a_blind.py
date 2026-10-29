"""No-model qualification of DETECTIVE-005A private blind rating custody."""
from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import detective_005a_assemble as assemble
from scripts import detective_005a_blind as blind
from scripts import detective_005a_fun_screen as screen
from scripts import detective_005a_live as live
from tests import test_detective_005a_fun_screen as fixtures


class Detective005ABlindCustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = screen.load_contract()

    def make_dataset(self, root: Path) -> Path:
        fixture = fixtures.Detective005AFunScreenTests()
        fixture.cfg = self.cfg
        rows = fixture.synthetic_rows()
        artifacts = root / "artifacts"
        for index in range(1, 13):
            lineage = f"G{index:02d}"
            folder = artifacts / lineage
            folder.mkdir(parents=True)
            members = {f"V-{lineage}-{s}" for s in "ABC"}
            selection = [r for r in rows if r["member_id"] in members]
            raw = folder / "results.jsonl"
            raw.write_text(
                "".join(screen.canonical(r) + "\n" for r in selection),
                encoding="utf-8")
            assemble.write_json(folder / "manifest.json", {
                "schema": "detective-005a-live-lineage-manifest-v1",
                "experiment": "DETECTIVE-005A",
                "issue": 176,
                "lineage": lineage,
                "source_plan": live.source_plan(lineage),
                "runner_script_sha256": screen.sha256_file(Path(live.__file__)),
                "results_jsonl_sha256": screen.sha256_file(raw),
                "results": 12,
                "members": [],
                "total_generated_tokens": 12,
                "peak_rss_kib": 1,
                "optimizer_steps": 0,
                "paid_api_calls": 0,
                "model_judge_calls": 0,
                "production_admission": False,
                "population_expansion_admitted": False,
            })
        dataset = root / "dataset"
        assemble.assemble(artifacts, dataset)
        return dataset

    def prepare_fixture(self, root: Path):
        dataset = self.make_dataset(root)
        reviewer, custody = root / "reviewer-only", root / "sealed-private"
        with mock.patch.object(blind.secrets, "token_bytes",
                               return_value=bytes(range(32))):
            result = blind.prepare(dataset, reviewer, custody)
        return dataset, reviewer, custody, result

    def filled_ratings(self, reviewer: Path, path: Path):
        ratings = screen.read_json(reviewer / "fun-ratings-template.json")
        ratings["reviewer_label"] = "external-human-demo"
        for record in ratings["ratings"]:
            record["score_0_to_4"] = 2
        path.write_bytes(blind.encoded(ratings))
        return ratings

    def test_public_fixed_salt_cannot_reconstruct_known_roster(self):
        fixture = fixtures.Detective005AFunScreenTests()
        fixture.cfg = self.cfg
        rows = fixture.synthetic_rows()
        with self.assertRaisesRegex(screen.ScreenError,
                                    "private 32-byte blinding key"):
            screen.build_review_packet(self.cfg, rows, b"public")
        first, map1 = screen.build_review_packet(
            self.cfg, rows, bytes(range(32)))
        second, map2 = screen.build_review_packet(
            self.cfg, rows, bytes(reversed(range(32))))
        self.assertNotEqual(map1, map2)
        self.assertNotEqual(first["private_mapping_commitment_sha256"],
                            second["private_mapping_commitment_sha256"])
        self.assertEqual(len(first["items"]), 144)
        self.assertFalse(first["identity_mapping_committed"])
        for member in self.cfg["source_population"]["members"]:
            self.assertNotIn(member["member_id"], screen.canonical(first))
        self.assertNotIn("secret_hex", screen.canonical(first))
        self.assertFalse(hasattr(screen, "REVIEW_SALT"))

    def test_private_prepare_separates_packet_and_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset, reviewer, custody, result = self.prepare_fixture(root)
            self.assertEqual(result["packet_items"], 144)
            self.assertFalse((dataset / "review").exists())
            self.assertFalse((reviewer / "private-custody.json").exists())
            self.assertFalse((custody / "fun-packet.json").exists())
            public = screen.read_json(reviewer / "fun-packet.json")
            private = screen.read_json(custody / "private-custody.json")
            self.assertEqual(private["packet_sha256"],
                             screen.sha256_file(reviewer / "fun-packet.json"))
            self.assertEqual(public["private_mapping_commitment_sha256"],
                             private["private_mapping_commitment_sha256"])
            self.assertEqual(len(private["mapping"]), 144)
            self.assertEqual(len(private["secret_hex"]), 64)
            self.assertEqual(len(public["items"]), 144)
            self.assertNotIn("V-G01-A", screen.canonical(public))
            self.assertNotIn(private["secret_hex"], screen.canonical(public))
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE((custody /
                                  "private-custody.json").stat().st_mode), 0o600)

    def test_seed_is_not_reused_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self.make_dataset(root)
            blind.prepare(dataset, root / "review1", root / "custody1")
            blind.prepare(dataset, root / "review2", root / "custody2")
            first = screen.read_json(root / "review1/fun-packet.json")
            second = screen.read_json(root / "review2/fun-packet.json")
            self.assertNotEqual(first["private_mapping_commitment_sha256"],
                                second["private_mapping_commitment_sha256"])
            self.assertNotEqual(
                screen.read_json(root / "custody1/private-custody.json")
                ["secret_hex"],
                screen.read_json(root / "custody2/private-custody.json")
                ["secret_hex"])

    def test_incomplete_or_forged_ratings_refused_before_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset, reviewer, custody, _ = self.prepare_fixture(root)
            empty = reviewer / "fun-ratings-template.json"
            with self.assertRaisesRegex(screen.ScreenError, "integer 0..4"):
                blind.freeze(dataset, reviewer, custody, empty, root / "frozen")
            self.assertFalse((root / "frozen").exists())
            rated_file = root / "completed-ratings.json"
            ratings = self.filled_ratings(reviewer, rated_file)
            altered = copy.deepcopy(ratings)
            altered["packet_sha256"] = "f" * 64
            rated_file.write_bytes(blind.encoded(altered))
            with self.assertRaisesRegex(screen.ScreenError,
                                        "rating packet identity/hash"):
                blind.freeze(dataset, reviewer, custody, rated_file,
                             root / "frozen")
            altered = copy.deepcopy(ratings)
            altered["ratings"][0]["item_id"] = "R999"
            rated_file.write_bytes(blind.encoded(altered))
            with self.assertRaisesRegex(screen.ScreenError, "rating order"):
                blind.freeze(dataset, reviewer, custody, rated_file,
                             root / "frozen")
            self.assertFalse((root / "frozen").exists())

    def test_packet_and_private_seed_tampering_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset, reviewer, custody, _ = self.prepare_fixture(root)
            private_path = custody / "private-custody.json"
            original = private_path.read_bytes()
            forged = json.loads(original)
            forged["secret_hex"] = ("a" * 64)
            private_path.write_bytes(blind.encoded(forged))
            with self.assertRaisesRegex(screen.ScreenError,
                                        "private mapping/packet commitment"):
                blind.authenticated_packet(dataset, reviewer, custody)
            private_path.write_bytes(original)
            public_path = reviewer / "fun-packet.json"
            packet_original = public_path.read_bytes()
            forged_packet = screen.read_json(public_path)
            forged_packet["items"][0]["response"] += " changed"
            public_path.write_bytes(blind.encoded(forged_packet))
            with self.assertRaisesRegex(screen.ScreenError,
                                        "private mapping/packet commitment"):
                blind.authenticated_packet(dataset, reviewer, custody)
            public_path.write_bytes(packet_original)

    def test_completed_ratings_sealed_before_explicit_reveal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset, reviewer, custody, _ = self.prepare_fixture(root)
            rating_file = root / "ratings.json"
            ratings = self.filled_ratings(reviewer, rating_file)
            frozen = root / "frozen"
            unblind = root / "unblinded"
            with self.assertRaisesRegex(screen.ScreenError, "unblinding"):
                blind.reveal(dataset, reviewer, custody, frozen, unblind, "")
            seal = blind.freeze(dataset, reviewer, custody,
                                rating_file, frozen)
            self.assertEqual(seal["ratings"], 144)
            self.assertFalse(seal["mapping_disclosed"])
            self.assertFalse((frozen / "revealed-item-map.json").exists())
            self.assertNotIn("V-G01-A", screen.canonical(
                screen.read_json(frozen / "freeze.json")))
            with self.assertRaisesRegex(screen.ScreenError, "unblinding"):
                blind.reveal(dataset, reviewer, custody, frozen, unblind,
                             "WRONG-TOKEN")
            output = blind.reveal(dataset, reviewer, custody, frozen, unblind,
                                  blind.REVEAL)
            self.assertTrue(output["unblinding_after_frozen_ratings"])
            self.assertEqual(output["members"], 36)
            self.assertEqual(output["shortlist"], 12)
            self.assertEqual(
                screen.read_json(unblind / "summary.json")["ratings_sha256"],
                screen.sha256_bytes(blind.encoded(ratings)))
            self.assertEqual(len(screen.read_json(
                unblind / "revealed-item-map.json")), 144)
            self.assertFalse(output["production_admission"])
            with self.assertRaisesRegex(screen.ScreenError, "refuse overwrite"):
                blind.reveal(dataset, reviewer, custody, frozen, unblind,
                             blind.REVEAL)

    def test_frozen_ratings_mutation_blocks_identity_disclosure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset, reviewer, custody, _ = self.prepare_fixture(root)
            rating_file = root / "ratings.json"
            self.filled_ratings(reviewer, rating_file)
            frozen = root / "frozen"
            blind.freeze(dataset, reviewer, custody, rating_file, frozen)
            ratings_path = frozen / "ratings.json"
            ratings_path.write_bytes(ratings_path.read_bytes() + b" ")
            with self.assertRaisesRegex(screen.ScreenError,
                                        "ratings are not sealed"):
                blind.reveal(dataset, reviewer, custody, frozen,
                             root / "unblinded", blind.REVEAL)
            self.assertFalse((root / "unblinded").exists())

    def test_custody_inside_repository_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self.make_dataset(root)
            with self.assertRaisesRegex(screen.ScreenError,
                                        "outside repository"):
                blind.prepare(dataset, screen.ROOT / "forbidden-review",
                              root / "custody")
            self.assertFalse((root / "custody").exists())
            with self.assertRaisesRegex(screen.ScreenError, "disjoint"):
                blind.prepare(dataset, root / "same", root / "same")
            with self.assertRaisesRegex(screen.ScreenError, "disjoint"):
                blind.prepare(dataset, root / "reviewer", dataset / "secret")
            with self.assertRaisesRegex(screen.ScreenError, "disjoint"):
                blind.prepare(dataset, dataset / "reviewer", root / "secret")

    def test_wsl_drvfs_ntfs_and_network_mounts_refuse_private_custody(self):
        # WSL reports os.name == "posix" even when /mnt/c uses an NTFS-
        # backed DrvFs mount; chmod cannot establish a private Windows ACL.
        if os.name != "posix":
            self.skipTest("POSIX-only mount security negative")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self.make_dataset(root)
            for filesystem in ("9p", "drvfs", "ntfs3", "fuseblk",
                               "nfs4", "cifs", "smb3"):
                with self.subTest(filesystem=filesystem):
                    with mock.patch.object(blind.sys, "platform", "linux"), \
                         mock.patch.object(blind, "_linux_mount_type",
                                           return_value=filesystem):
                        with self.assertRaisesRegex(
                                screen.ScreenError, "vetted local filesystem"):
                            blind.prepare(dataset, root / "review",
                                          root / "custody")
                    self.assertFalse((root / "review").exists())
                    self.assertFalse((root / "custody").exists())

    def test_mountinfo_uses_deepest_matching_mount(self):
        if os.name != "posix":
            self.skipTest("Linux-only mount qualification")
        example = (
            "21 0 0:1 / / rw - overlay overlay rw\n"
            "22 21 0:2 / /mnt/c rw - 9p C: rw,aname=drvfs\n"
            "23 22 0:3 / /mnt/c/secrets rw - ntfs3 /dev/sda rw\n"
        )
        with mock.patch.object(blind.Path, "read_text", return_value=example):
            self.assertEqual(blind._linux_mount_type(
                Path("/mnt/c/secrets/owner")), "ntfs3")
            self.assertEqual(blind._linux_mount_type(
                Path("/mnt/c/owner")), "9p")
            self.assertEqual(blind._linux_mount_type(
                Path("/home/owner")), "overlay")

    def test_unverified_windows_acl_custody_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self.make_dataset(root)
            with mock.patch.object(blind, "secure_custody_available",
                                   return_value=False):
                with self.assertRaisesRegex(screen.ScreenError,
                                            "requires POSIX permissions"):
                    blind.prepare(dataset, root / "review", root / "custody")
            self.assertFalse((root / "review").exists())
            self.assertFalse((root / "custody").exists())

    def test_public_archive_fails_closed_if_review_packet_added(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self.make_dataset(root)
            (dataset / "review").mkdir()
            with self.assertRaisesRegex(screen.ScreenError,
                                        "public dataset must not include"):
                assemble.verify_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
