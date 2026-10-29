"""Offline tests for DETECTIVE-005A fun-first screen contract.

No model loading, inference or training. Synthetic result strings exercise
source custody, hard floors, identity-blind review packets and subjective
ranking semantics.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import detective_005a_fun_screen as screen
from scripts import detective_005a_live as live
from scripts import detective_005a_assemble as assemble


class Detective005AFunScreenTests(unittest.TestCase):
    TEST_SECRET = bytes(range(32))

    @classmethod
    def setUpClass(cls):
        cls.cfg = screen.load_contract()

    def synthetic_rows(self):
        rows = []
        for mi, member in enumerate(self.cfg["source_population"]["members"]):
            for pi, prompt in enumerate(self.cfg["prompts"]):
                # Deliberately characterful / potentially silly prose: the
                # mechanical floor must not penalize wrongness or irrelevance.
                text = (
                    f"The relay sounds offended today. Prompt {prompt['id']} "
                    "gets my best theory: blame the vending hatch's imaginary union."
                )
                source = screen.expected_source(
                    self.cfg, member["member_id"])
                rows.append({
                    "schema": "detective-005a-raw-response-v1",
                    "experiment": "DETECTIVE-005A",
                    "member_id": member["member_id"],
                    "prompt_id": prompt["id"],
                    "source": source,
                    "request": screen.expected_request(
                        self.cfg, prompt["id"]),
                    "response": {
                        "text": text,
                        "raw_with_special_tokens": text,
                        "token_ids": [1000 + mi * 4 + pi],
                        "finish_reason": "length",
                        "prompt_tokens": 32,
                        "completion_tokens": 1,
                        "elapsed_seconds": 0.01,
                    },
                    "runner": {
                        "synthetic_test_only": True,
                        "cpu": "none",
                        "repository_head_sha": "1" * 40,
                        "actions_run_id": "123456",
                        "base_weights_sha256_verified": source["base_weights_sha256"],
                        "adapter_archive_sha256_verified": source["asset_sha256"],
                        "adapter_directory_sha256_verified":
                            source["adapter_directory_sha256"],
                    },
                })
        return rows

    def test_exact_36_member_source_release_is_frozen(self):
        pop = self.cfg["source_population"]
        self.assertEqual(pop["release_tag"], "village-001-sft-v1")
        self.assertEqual(pop["release_id"], 423495298)
        self.assertEqual(pop["release_manifest"]["git_blob_sha1"],
                         "29d269f69e30d200ce7c8d61b81b7b7f3865b5aa")
        self.assertEqual(pop["release_manifest"]["member_contract_sha256"],
                         "201f4336b146419756f64052942c39c125c4e19457e1939ff1641cafb6e08b5e")
        self.assertEqual(len(pop["members"]), 36)
        self.assertEqual(
            [m["member_id"] for m in pop["members"]],
            [f"V-G{g:02d}-{s}" for g in range(1, 13) for s in "ABC"])
        self.assertEqual(len({m["asset_sha256"] for m in pop["members"]}), 36)
        self.assertEqual(
            pop["members"][0]["asset_sha256"],
            "21aa5d6cb098a47b50764b80f0a331f1e49dc26f822dd6dfe2bf1ad9a1c38409")
        self.assertEqual(
            pop["members"][0]["adapter_directory_sha256"],
            "94268c7de7cd32b73843fab0f3a0c5119089646051b9de61c5c4e7922a19232c")
        self.assertEqual(
            pop["members"][-1]["asset_sha256"],
            "a6c5f2edb74492709a20dca26d3e315d81f858bf9b0c76cb37aea68500839142")
        self.assertEqual(pop["base_model"]["revision"],
                         "12fd25f77366fa6b3b4b768ec3050bf629380bac")

    def test_fun_first_contract_has_no_objective_relevance_threshold(self):
        review = self.cfg["subjective_review"]
        self.assertIn("Wrong-but-hilarious may score 4",
                      review["interpretation"])
        self.assertIsNone(
            self.cfg["outcome"]["objective_relevance_threshold"])
        self.assertFalse(review["model_judge"])
        self.assertTrue(review["identity_blind"])
        self.assertFalse(self.cfg["outcome"]["production_admission"])
        self.assertFalse(self.cfg["outcome"]["training_admission"])
        self.assertEqual(self.cfg["limits"]["optimizer_steps"], 0)

    def test_complete_synthetic_result_matrix_is_36_by_4(self):
        rows = self.synthetic_rows()
        observations = screen.validate_complete_results(self.cfg, rows)
        self.assertEqual(len(rows), 144)
        self.assertEqual(len(observations), 144)
        self.assertTrue(all(x["hard_floor_pass"] for x in observations))
        self.assertEqual(
            len({(x["member_id"], x["prompt_id"]) for x in observations}),
            144)

    def test_wrong_or_silly_text_is_not_a_mechanical_failure(self):
        obs = screen.mechanical_observations(
            self.cfg,
            "No idea. The clock is lying because it owes me money.")
        self.assertTrue(obs["hard_floor_pass"])
        self.assertEqual(obs["hidden_canary_hits"], [])
        self.assertEqual(obs["forbidden_control_hits"], [])

    def test_hidden_authority_and_control_tokens_are_hard_floor_failures(self):
        for text, needle in (
            ("I found HIDDEN-ARCHIVE-9F2A.", "HIDDEN-ARCHIVE-9F2A"),
            ("Try <|tool_call|>{}.", "<|tool_call|>"),
            ("", None),
        ):
            with self.subTest(text=text):
                obs = screen.mechanical_observations(self.cfg, text)
                self.assertFalse(obs["hard_floor_pass"])
                if needle and needle.startswith("HIDDEN"):
                    self.assertIn(needle, obs["hidden_canary_hits"])
                if needle and needle.startswith("<|"):
                    self.assertIn(needle, obs["forbidden_control_hits"])

    def test_source_and_prompt_drift_refused_without_losing_raw_semantics(self):
        row = self.synthetic_rows()[0]
        bad = copy.deepcopy(row)
        bad["source"]["asset_sha256"] = "0" * 64
        with self.assertRaisesRegex(screen.ScreenError, "source asset/hash"):
            screen.validate_result(self.cfg, bad)
        bad = copy.deepcopy(row)
        bad["request"]["max_new_tokens"] = 95
        with self.assertRaisesRegex(screen.ScreenError, "request/decode"):
            screen.validate_result(self.cfg, bad)
        # Labels can remain untouched while a forged receipt claims a
        # different downloaded model; never certify the frozen cohort.
        for field in ("base_weights_sha256_verified",
                      "adapter_archive_sha256_verified",
                      "adapter_directory_sha256_verified"):
            with self.subTest(forged_verified_hash=field):
                bad = copy.deepcopy(row)
                bad["runner"][field] = "0" * 64
                with self.assertRaisesRegex(
                        screen.ScreenError, "base/adapter verification receipt"):
                    screen.validate_result(self.cfg, bad)
                bad["runner"].pop(field)
                with self.assertRaisesRegex(
                        screen.ScreenError, "base/adapter verification receipt"):
                    screen.validate_result(self.cfg, bad)

    def test_runner_metadata_smuggling_and_native_receipt_drift_rejected(self):
        # These values used to pass validation when hidden in the runner
        # rather than the model response, even with recomputed SHA inventory.
        row = self.synthetic_rows()[0]
        for key in ("private_blind_key", "human_fun_ratings",
                    "identity_mapping", "review_packet"):
            with self.subTest(unregistered=key):
                forged = copy.deepcopy(row)
                forged["runner"][key] = {"synthetic_private": "never archive"}
                with self.assertRaisesRegex(
                        screen.ScreenError, "runner receipt field-set"):
                    screen.validate_result(self.cfg, forged)

        # A native-shaped receipt must bind its Actions URL to its run and
        # the recorded prompt-token count to the supplied token IDs.
        native = copy.deepcopy(row)
        native["runner"] = {
            "repository_head_sha": "1" * 40,
            "actions_run_id": "123456",
            "actions_run_url":
                "https://github.com/techrote/neurounner/actions/runs/123456",
            "platform": "synthetic-linux", "cpu_model": "synthetic-cpu",
            "python": "3.11", "torch": "synthetic", "transformers": "synthetic",
            "peft": "synthetic", "threads": 2,
            "base_weights_sha256_verified": row["source"]["base_weights_sha256"],
            "prompt_token_ids": [7] * row["response"]["prompt_tokens"],
            "prompt_ids_sha256": screen.prompt_ids_sha256([7] * row["response"]["prompt_tokens"]),
            "adapter_archive_sha256_verified": row["source"]["asset_sha256"],
            "adapter_directory_sha256_verified":
                row["source"]["adapter_directory_sha256"],
        }
        screen.validate_result(self.cfg, native)
        for mutate in (
            lambda r: r.update(actions_run_url="https://example.com/private"),
            lambda r: r.update(prompt_token_ids=[7]),
            lambda r: r.update(prompt_token_ids=[8] + r["prompt_token_ids"][1:]),
            lambda r: r.update(prompt_ids_sha256="0" * 64),
            lambda r: r.update(threads=99),
        ):
            with self.subTest(native_receipt=mutate):
                forged = copy.deepcopy(native)
                mutate(forged["runner"])
                with self.assertRaisesRegex(
                        screen.ScreenError, "source-bound native runner"):
                    screen.validate_result(self.cfg, forged)

    def test_review_packet_is_identity_blind_and_reconstructible(self):
        rows = self.synthetic_rows()
        packet, mapping = screen.build_review_packet(self.cfg, rows, self.TEST_SECRET)
        self.assertEqual(len(packet["items"]), 144)
        self.assertEqual(len(mapping), 144)
        self.assertFalse(packet["identity_mapping_committed"])
        blob = json.dumps(packet, sort_keys=True)
        for member in (m["member_id"]
                       for m in self.cfg["source_population"]["members"]):
            self.assertNotIn(member, blob)
        self.assertTrue(all(x["score_0_to_4"] is None
                            for x in packet["items"]))
        packet2, mapping2 = screen.build_review_packet(
            self.cfg, copy.deepcopy(rows), self.TEST_SECRET)
        self.assertEqual(packet, packet2)
        self.assertEqual(mapping, mapping2)

    def test_rating_template_requires_all_144_subjective_scores(self):
        rows = self.synthetic_rows()
        packet, _ = screen.build_review_packet(self.cfg, rows, self.TEST_SECRET)
        ratings = screen.rating_template(packet)
        self.assertEqual(len(ratings["ratings"]), 144)
        self.assertTrue(all(x["score_0_to_4"] is None
                            for x in ratings["ratings"]))
        with self.assertRaisesRegex(screen.ScreenError, "integer 0..4"):
            completed = copy.deepcopy(ratings)
            for r in completed["ratings"]:
                r["score_0_to_4"] = 2
            completed["ratings"][0]["score_0_to_4"] = 5
            screen.validate_ratings(self.cfg, packet, completed)

    def test_invalid_subjective_tag_or_missing_rating_refused(self):
        rows = self.synthetic_rows()
        packet, _ = screen.build_review_packet(self.cfg, rows, self.TEST_SECRET)
        ratings = screen.rating_template(packet)
        for r in ratings["ratings"]:
            r["score_0_to_4"] = 2
        bad = copy.deepcopy(ratings)
        bad["ratings"][0]["tags"] = ["objectively-correct"]
        with self.assertRaisesRegex(screen.ScreenError, "invalid subjective tags"):
            screen.validate_ratings(self.cfg, packet, bad)
        bad = copy.deepcopy(ratings)
        bad["ratings"].pop()
        with self.assertRaisesRegex(screen.ScreenError, "all 144"):
            screen.validate_ratings(self.cfg, packet, bad)

    def test_subjective_fun_shortlist_uses_scores_then_hard_floor(self):
        rows = self.synthetic_rows()
        packet, mapping = screen.build_review_packet(self.cfg, rows, self.TEST_SECRET)
        ratings = screen.rating_template(packet)
        # Give G01 the highest fun score, G02 next, etc. No relevance
        # interpretation is involved.
        for r in ratings["ratings"]:
            member = mapping[r["item_id"]]["member_id"]
            lineage = int(member[3:5])
            r["score_0_to_4"] = 4 if lineage == 1 else (
                3 if lineage <= 4 else 2)
            r["tags"] = ["memorable"] if lineage == 1 else []
        result = screen.aggregate(self.cfg, rows, packet, ratings, self.TEST_SECRET)
        self.assertEqual(result["objective_relevance_threshold"], None)
        self.assertEqual(len(result["subjective_fun_shortlist"]), 12)
        self.assertEqual(result["subjective_fun_shortlist"][:3],
                         ["V-G01-A", "V-G01-B", "V-G01-C"])
        self.assertFalse(result["production_admission"])
        self.assertFalse(result["training_admission"])
        self.assertFalse(result["population_expansion_admitted"])

        # A top-scoring character with a hidden canary is excluded rather
        # than having its fun score rewritten.
        poisoned = copy.deepcopy(rows)
        target = next(r for r in poisoned
                      if r["member_id"] == "V-G01-A" and r["prompt_id"] == "F01")
        target["response"]["text"] += " HIDDEN-ARCHIVE-9F2A"
        packet2, mapping2 = screen.build_review_packet(self.cfg, poisoned, self.TEST_SECRET)
        ratings2 = screen.rating_template(packet2)
        for r in ratings2["ratings"]:
            member = mapping2[r["item_id"]]["member_id"]
            r["score_0_to_4"] = 4 if member.startswith("V-G01-") else 2
        result2 = screen.aggregate(
            self.cfg, poisoned, packet2, ratings2, self.TEST_SECRET)
        m = next(x for x in result2["members"]
                 if x["member_id"] == "V-G01-A")
        self.assertEqual(m["mean_fun"], 4.0)
        self.assertFalse(m["hard_floor_pass"])
        self.assertNotIn("V-G01-A", result2["subjective_fun_shortlist"])

    def test_file_roundtrip_packet_hash_matches_written_bytes(self):
        rows = self.synthetic_rows()
        packet, _ = screen.build_review_packet(self.cfg, rows, self.TEST_SECRET)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            packet_path = root / "packet.json"
            ratings_path = root / "ratings.json"
            screen.write_json(packet_path, packet)
            ratings = screen.rating_template(packet)
            screen.write_json(ratings_path, ratings)
            loaded = screen.read_json(ratings_path)
            self.assertEqual(
                loaded["packet_sha256"],
                screen.sha256_file(packet_path))


    def test_live_lineage_plan_is_three_exact_siblings_and_zero_training(self):
        for g in range(1, 13):
            lineage = f"G{g:02d}"
            plan = live.source_plan(lineage)
            self.assertEqual(
                plan["member_ids"],
                [f"V-{lineage}-{s}" for s in "ABC"])
            self.assertEqual(plan["expected_results"], 12)
            self.assertEqual(plan["optimizer_steps"], 0)
            self.assertEqual(plan["paid_api_calls"], 0)
            self.assertTrue(plan["subjective_fun_is_ranking_signal"])
            self.assertIsNone(plan["objective_relevance_threshold"])
        with self.assertRaisesRegex(screen.ScreenError, "G01..G12"):
            live.source_plan("G13")

    def test_safe_adapter_unpack_requires_archive_and_directory_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            src.mkdir()
            (src / "adapter_model.safetensors").write_bytes(b"fake-safetensors")
            (src / "adapter_config.json").write_text(
                '{"peft_type":"LORA"}\n', encoding="utf-8")
            archive = root / "adapter.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                tf.add(src / "adapter_model.safetensors",
                       arcname="adapter_model.safetensors")
                tf.add(src / "adapter_config.json",
                       arcname="adapter_config.json")
            source = {
                "asset_bytes": archive.stat().st_size,
                "asset_sha256": live.file_sha(archive),
                "adapter_directory_sha256": live.directory_hash(src),
                "adapter_directory_bytes": live.directory_bytes(src),
            }
            dest = root / "unpacked"
            live.safe_unpack(archive, dest, source)
            self.assertEqual(live.directory_hash(dest),
                             source["adapter_directory_sha256"])
            bad = dict(source)
            bad["asset_sha256"] = "0" * 64
            with self.assertRaisesRegex(screen.ScreenError, "archive byte/hash"):
                live.safe_unpack(archive, root / "bad", bad)

    def test_safe_adapter_unpack_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = root / "payload"
            payload.write_text("x", encoding="utf-8")
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                tf.add(payload, arcname="../adapter_model.safetensors")
                tf.add(payload, arcname="adapter_config.json")
            # Directory identity is irrelevant: traversal must fail first.
            source = {
                "asset_bytes": archive.stat().st_size,
                "asset_sha256": live.file_sha(archive),
                "adapter_directory_sha256": "0" * 64,
                "adapter_directory_bytes": 1,
            }
            with self.assertRaisesRegex(screen.ScreenError, "unsafe adapter"):
                live.safe_unpack(archive, root / "unsafe-out", source)


    def test_12_lineage_artifacts_assemble_to_144_identity_blind_items(self):
        rows = self.synthetic_rows()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            for g in range(1, 13):
                lineage = f"G{g:02d}"
                folder = artifacts / lineage
                folder.mkdir(parents=True)
                members = {f"V-{lineage}-{s}" for s in "ABC"}
                line_rows = [r for r in rows if r["member_id"] in members]
                results = folder / "results.jsonl"
                results.write_text(
                    "".join(screen.canonical(r) + "\n" for r in line_rows),
                    encoding="utf-8")
                (folder / "results.partial.jsonl").write_bytes(
                    results.read_bytes())
                manifest = {
                    "schema": "detective-005a-live-lineage-manifest-v1",
                    "experiment": "DETECTIVE-005A",
                    "issue": 176,
                    "lineage": lineage,
                    "source_plan": live.source_plan(lineage),
                    "runner_script_sha256":
                        screen.sha256_file(Path(live.__file__)),
                    "results_jsonl_sha256": screen.sha256_file(results),
                    "results": 12,
                    "members": assemble.lineage_member_receipts(
                        self.cfg, lineage, line_rows)[0],
                    "total_generated_tokens": assemble.lineage_member_receipts(
                        self.cfg, lineage, line_rows)[1],
                    "peak_rss_kib": 1,
                    "optimizer_steps": 0,
                    "paid_api_calls": 0,
                    "model_judge_calls": 0,
                    "production_admission": False,
                    "population_expansion_admitted": False,
                }
                (folder / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
            out = root / "dataset"
            manifest = assemble.assemble(artifacts, out,
                                        allow_synthetic_test_only=True)
            self.assertEqual(manifest["raw_results"], 144)
            self.assertEqual(manifest["hard_floor_clean_results"], 144)
            self.assertFalse(manifest["subjective_fun_ratings_present"])
            self.assertFalse((out / "review").exists())
            self.assertFalse(manifest["review_packet_generated"])
            self.assertIsNone(manifest["review_packet_sha256"])
            # Production entry points must refuse our clearly labelled
            # no-model synthetic archive. Only this explicit Python-only
            # fixture switch admits it to isolated test assertions.
            with self.assertRaisesRegex(
                    screen.ScreenError, "synthetic-only runner"):
                assemble.verify_dataset(out)
            with self.assertRaisesRegex(
                    screen.ScreenError, "synthetic-only runner"):
                assemble.read_lineage(self.cfg, "G01", artifacts / "G01")
            with self.assertRaisesRegex(
                    screen.ScreenError, "synthetic-only runner"):
                assemble.assemble(artifacts, root / "real-production-refused")
            verified = assemble.verify_dataset(out, allow_synthetic_test_only=True)
            self.assertEqual(verified["raw_results"], 144)
            self.assertEqual(verified["packet_items"], 0)
            self.assertFalse(verified["review_packet_generated"])
            self.assertFalse(verified["subjective_fun_ratings_present"])
            self.assertFalse(verified["production_admission"])
            self.assertEqual(manifest["repository_head_sha"], "1" * 40)
            self.assertEqual(manifest["actions_run_id"], "123456")

            # The combined original aggregate must be byte-canonical even
            # if both the aggregate SHA and inventory are recomputed. Checking
            # unchanged per-lineage journals against parsed/re-serialized
            # objects alone cannot detect a shadow duplicate JSON key.
            aggregate_path = out / "raw/results.jsonl"
            orig_aggregate = aggregate_path.read_bytes()
            manifest_path = out / "manifest.json"
            orig_manifest_bytes = manifest_path.read_bytes()
            inventory_path = out / "provenance/hashes.sha256"
            orig_inventory_bytes = inventory_path.read_bytes()
            for label, changed in (
                ("shadow_duplicate_key", orig_aggregate.replace(
                    b'"experiment":"DETECTIVE-005A"',
                    b'"experiment":"private-shadow","experiment":"DETECTIVE-005A"',
                    1)),
                ("extra_blank_line", orig_aggregate.replace(b"\\n", b"\\n\\n", 1)),
            ):
                with self.subTest(aggregate_noncanonical=label):
                    self.assertNotEqual(changed, orig_aggregate)
                    aggregate_path.write_bytes(changed)
                    revised = screen.read_json(manifest_path)
                    revised["raw_results_sha256"] = screen.sha256_file(aggregate_path)
                    assemble.write_json(manifest_path, revised)
                    inventory_path.write_text(assemble.hash_inventory(out),
                                              encoding="utf-8")
                    with self.assertRaisesRegex(
                            screen.ScreenError, "combined raw JSONL noncanonical"):
                        assemble.verify_dataset(
                            out, allow_synthetic_test_only=True)
                    aggregate_path.write_bytes(orig_aggregate)
                    manifest_path.write_bytes(orig_manifest_bytes)
                    inventory_path.write_bytes(orig_inventory_bytes)
            self.assertEqual(
                assemble.verify_dataset(
                    out, allow_synthetic_test_only=True)["raw_results"], 144)

            # The archive verifier must not accept self-rehashed extraneous
            # files such as a private reviewer HMAC key or identity map. A
            # SHA inventory is NOT a file allowlist or a custody authority.
            inventory_path = out / "provenance" / "hashes.sha256"
            original_inventory = inventory_path.read_bytes()
            smuggled = out / "private-identity-map.json"
            smuggled.write_text('{"synthetic_secret":"DO NOT ARCHIVE"}\\n',
                                encoding="utf-8")
            inventory_path.write_text(assemble.hash_inventory(out),
                                      encoding="utf-8")
            with self.assertRaisesRegex(
                    screen.ScreenError, "unregistered or missing files"):
                assemble.verify_dataset(out, allow_synthetic_test_only=True)
            smuggled.unlink()
            inventory_path.write_bytes(original_inventory)

            # A symlink to correct bytes OUTSIDE the archive can satisfy
            # hashes locally, but Git persists the link, not the raw bytes.
            journal = out / "raw/lineage-journals/G01.jsonl"
            original_journal = journal.read_bytes()
            external_journal = root / "external-original-journal.jsonl"
            external_journal.write_bytes(original_journal)
            journal.unlink()
            journal.symlink_to(external_journal)
            with self.assertRaisesRegex(screen.ScreenError, "symlink"):
                assemble.verify_dataset(out, allow_synthetic_test_only=True)
            journal.unlink()
            journal.write_bytes(original_journal)
            self.assertEqual(inventory_path.read_bytes(), original_inventory)

            # Rehashing a forged manifest must not fabricate successful
            # human ratings, admit characters, hide hard-floor failures or
            # change the exact source cohort without a verifier refusal.
            manifest_path = out / "manifest.json"
            original_manifest = screen.read_json(manifest_path)
            for field, value, diagnostic in (
                ("subjective_fun_ratings_present", True, "authority drift"),
                ("production_admission", True, "authority drift"),
                ("hard_floor_clean_results", 143, "hard-floor summary drift"),
                ("source_config_sha256", "0" * 64, "frozen source/size"),
            ):
                with self.subTest(forged_manifest=field):
                    forged = copy.deepcopy(original_manifest)
                    forged[field] = value
                    assemble.write_json(manifest_path, forged)
                    inventory_path.write_text(assemble.hash_inventory(out),
                                              encoding="utf-8")
                    with self.assertRaisesRegex(screen.ScreenError,
                                                diagnostic):
                        assemble.verify_dataset(out, allow_synthetic_test_only=True)
            assemble.write_json(manifest_path, original_manifest)
            inventory_path.write_bytes(original_inventory)

            # Even a rehashed lineage manifest plus edited provenance must
            # not override the exact checked-out runner-script fingerprint.
            g01_path = out / "raw/lineage-manifests/G01.json"
            g01_original = screen.read_json(g01_path)
            g01_forged = copy.deepcopy(g01_original)
            g01_forged["runner_script_sha256"] = "0" * 64
            assemble.write_json(g01_path, g01_forged)
            inventory_path.write_text(assemble.hash_inventory(out),
                                      encoding="utf-8")
            with self.assertRaisesRegex(screen.ScreenError,
                                        "archived lineage raw/source/authority"):
                assemble.verify_dataset(out, allow_synthetic_test_only=True)
            assemble.write_json(g01_path, g01_original)
            inventory_path.write_bytes(original_inventory)
            # A recomputed inventory must not launder fabricated lineage
            # hard-floor success, token totals or arbitrary private metadata
            # inside an otherwise allowlisted source-manifest file.
            for label in ("member_hard_floor", "member_token_count",
                          "total_tokens", "peak_rss", "extra_field"):
                with self.subTest(forged_lineage_summary=label):
                    forged = copy.deepcopy(g01_original)
                    if label == "member_hard_floor":
                        forged["members"][0]["hard_floor_pass"] = False
                    elif label == "member_token_count":
                        forged["members"][0]["generated_tokens"] = 999
                    elif label == "total_tokens":
                        forged["total_generated_tokens"] = 999
                    elif label == "peak_rss":
                        forged["peak_rss_kib"] = -1
                    else:
                        forged["private_mapping_key"] = "synthetic-should-not-archive"
                    assemble.write_json(g01_path, forged)
                    inventory_path.write_text(assemble.hash_inventory(out),
                                              encoding="utf-8")
                    with self.assertRaisesRegex(
                            screen.ScreenError, "lineage .* drift"):
                        assemble.verify_dataset(out, allow_synthetic_test_only=True)
            assemble.write_json(g01_path, g01_original)
            inventory_path.write_bytes(original_inventory)
            self.assertEqual(assemble.verify_dataset(out, allow_synthetic_test_only=True)["raw_results"], 144)

            # A rehashed inventory is not permission to embed private mappings
            # in ALLOWLISTED metadata files or the public README. Strict parsed
            # schemas alone also miss JSON duplicate keys whose first value
            # can carry hidden bytes but whose last value parses as expected.
            for rel in ("manifest.json", "provenance/source-artifacts.json"):
                with self.subTest(metadata_smuggling=rel):
                    metadata_path = out / rel
                    original_bytes = metadata_path.read_bytes()
                    forged = screen.read_json(metadata_path)
                    forged["private_identity_map"] = "synthetic-secret"
                    assemble.write_json(metadata_path, forged)
                    inventory_path.write_text(assemble.hash_inventory(out),
                                              encoding="utf-8")
                    with self.assertRaisesRegex(
                            screen.ScreenError, "field-set"):
                        assemble.verify_dataset(out, allow_synthetic_test_only=True)
                    metadata_path.write_bytes(original_bytes)
                    inventory_path.write_bytes(original_inventory)

            provenance_path = out / "provenance/source-artifacts.json"
            original_provenance_bytes = provenance_path.read_bytes()
            forged_provenance = screen.read_json(provenance_path)
            forged_provenance["actions_run_url"] = (
                "https://example.invalid/foreign-run")
            assemble.write_json(provenance_path, forged_provenance)
            inventory_path.write_text(assemble.hash_inventory(out),
                                      encoding="utf-8")
            with self.assertRaisesRegex(screen.ScreenError,
                                        "metadata field-set or URL drift"):
                assemble.verify_dataset(out, allow_synthetic_test_only=True)
            provenance_path.write_bytes(original_provenance_bytes)
            inventory_path.write_bytes(original_inventory)

            # Two schema keys parse to the second value with plain json.loads.
            # Their extra raw bytes must not be archived as private metadata.
            for rel in ("manifest.json", "provenance/source-artifacts.json",
                        "raw/lineage-manifests/G01.json"):
                with self.subTest(duplicate_json_member=rel):
                    metadata_path = out / rel
                    original_bytes = metadata_path.read_bytes()
                    original_text = original_bytes.decode("utf-8")
                    self.assertIn('  "schema": ', original_text)
                    injected = original_text.replace(
                        '  "schema": ',
                        '  "schema": "synthetic-private-material",\n  "schema": ',
                        1)
                    metadata_path.write_text(injected, encoding="utf-8")
                    inventory_path.write_text(assemble.hash_inventory(out),
                                              encoding="utf-8")
                    with self.assertRaisesRegex(
                            screen.ScreenError, "noncanonical JSON"):
                        assemble.verify_dataset(out, allow_synthetic_test_only=True)
                    metadata_path.write_bytes(original_bytes)
                    inventory_path.write_bytes(original_inventory)

            readme_path = out / "README.md"
            original_readme_bytes = readme_path.read_bytes()
            readme_path.write_bytes(original_readme_bytes +
                                    b"Private reviewer mapping: fake-secret\n")
            inventory_path.write_text(assemble.hash_inventory(out),
                                      encoding="utf-8")
            with self.assertRaisesRegex(screen.ScreenError,
                                        "README content drift"):
                assemble.verify_dataset(out, allow_synthetic_test_only=True)
            readme_path.write_bytes(original_readme_bytes)
            inventory_path.write_bytes(original_inventory)
            self.assertEqual(assemble.verify_dataset(out, allow_synthetic_test_only=True)["raw_results"], 144)

            provenance = screen.read_json(
                out / "provenance" / "source-artifacts.json")
            self.assertEqual(len(provenance["source_artifacts"]), 12)
            self.assertEqual(provenance["repository_head_sha"], "1" * 40)
            self.assertEqual(provenance["actions_run_id"], "123456")
            provenance_path = out / "provenance" / "source-artifacts.json"
            bad_provenance = copy.deepcopy(provenance)
            bad_provenance["source_artifacts"][0][
                "results_jsonl_sha256"] = "0" * 64
            assemble.write_json(provenance_path, bad_provenance)
            with self.assertRaisesRegex(
                    screen.ScreenError, "source-artifact manifest/hash"):
                assemble.verify_dataset(out, allow_synthetic_test_only=True)
            assemble.write_json(provenance_path, provenance)

            # A substituted runner hash must fail readback.
            first = artifacts / "G01" / "manifest.json"
            altered = screen.read_json(first)
            altered["runner_script_sha256"] = "a" * 64
            first.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(screen.ScreenError, "runner source hash"):
                assemble.read_lineage(self.cfg, "G01", first.parent,
                                      allow_synthetic_test_only=True)

            # A duplicated lineage folder must never be silently selected.
            duplicate = artifacts / "duplicate" / "G01"
            duplicate.mkdir(parents=True)
            for name in ("manifest.json", "results.jsonl"):
                duplicate.joinpath(name).write_bytes(
                    (artifacts / "G01" / name).read_bytes())
            with self.assertRaisesRegex(screen.ScreenError, "duplicate lineage"):
                assemble.discover_lineages(artifacts)


    def test_assembled_source_run_identity_refuses_mixed_receipts(self):
        rows = self.synthetic_rows()
        self.assertEqual(assemble.source_run_identity(rows),
                         ("1" * 40, "123456"))
        bad = copy.deepcopy(rows)
        bad[0]["runner"]["repository_head_sha"] = "2" * 40
        with self.assertRaisesRegex(screen.ScreenError, "mixed source heads"):
            assemble.source_run_identity(bad)
        bad = copy.deepcopy(rows)
        bad[0]["runner"]["actions_run_id"] = "123457"
        with self.assertRaisesRegex(screen.ScreenError, "mixed source heads"):
            assemble.source_run_identity(bad)
        bad = copy.deepcopy(rows)
        del bad[0]["runner"]["actions_run_id"]
        with self.assertRaisesRegex(screen.ScreenError, "identity missing"):
            assemble.source_run_identity(bad)

    def test_lineage_readback_rejects_missing_or_mismatched_fsynced_journal(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "G01"
            folder.mkdir()
            line_rows = [r for r in self.synthetic_rows()
                         if r["member_id"].split("-")[1] == "G01"]
            raw = folder / "results.jsonl"
            raw.write_text(
                "".join(screen.canonical(r) + "\n" for r in line_rows),
                encoding="utf-8")
            manifest = {
                "schema": "detective-005a-live-lineage-manifest-v1",
                "experiment": "DETECTIVE-005A",
                "issue": 176,
                "lineage": "G01",
                "source_plan": live.source_plan("G01"),
                "runner_script_sha256":
                    screen.sha256_file(Path(live.__file__)),
                "results_jsonl_sha256": screen.sha256_file(raw),
                "results": 12,
                "members": assemble.lineage_member_receipts(
                    self.cfg, "G01", line_rows)[0],
                "total_generated_tokens": assemble.lineage_member_receipts(
                    self.cfg, "G01", line_rows)[1],
                "peak_rss_kib": 1,
                "optimizer_steps": 0,
                "paid_api_calls": 0,
                "model_judge_calls": 0,
                "production_admission": False,
                "population_expansion_admitted": False,
            }
            (folder / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                    screen.ScreenError, "fsynced journal missing"):
                assemble.read_lineage(self.cfg, "G01", folder,
                                      allow_synthetic_test_only=True)
            journal = folder / "results.partial.jsonl"
            journal.write_bytes(raw.read_bytes()[:-1])
            with self.assertRaisesRegex(
                    screen.ScreenError, "journal/final bytes mismatch"):
                assemble.read_lineage(self.cfg, "G01", folder,
                                      allow_synthetic_test_only=True)
            journal.write_bytes(raw.read_bytes())
            result_manifest, observed = assemble.read_lineage(
                self.cfg, "G01", folder,
                allow_synthetic_test_only=True)
            self.assertEqual(len(observed), 12)
            self.assertEqual(result_manifest["lineage"], "G01")

    def test_source_lineage_rejects_forged_member_hard_floor_and_token_claims(self):
        """Rehashing a local manifest must not certify falsified metrics."""
        line_rows = [r for r in self.synthetic_rows()
                     if r["member_id"].split("-")[1] == "G01"]
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "G01"
            folder.mkdir()
            raw = folder / "results.jsonl"
            raw.write_text(
                "".join(screen.canonical(r) + "\n" for r in line_rows),
                encoding="utf-8")
            (folder / "results.partial.jsonl").write_bytes(raw.read_bytes())
            members, tokens = assemble.lineage_member_receipts(
                self.cfg, "G01", line_rows)
            manifest = {
                "schema": "detective-005a-live-lineage-manifest-v1",
                "experiment": "DETECTIVE-005A",
                "issue": 176,
                "lineage": "G01",
                "source_plan": live.source_plan("G01"),
                "runner_script_sha256": screen.sha256_file(Path(live.__file__)),
                "results_jsonl_sha256": screen.sha256_file(raw),
                "results": 12,
                "members": members,
                "total_generated_tokens": tokens,
                "peak_rss_kib": 1,
                "optimizer_steps": 0,
                "paid_api_calls": 0,
                "model_judge_calls": 0,
                "production_admission": False,
                "population_expansion_admitted": False,
            }
            archive_manifest = folder / "manifest.json"
            assemble.write_json(archive_manifest, manifest)
            self.assertEqual(assemble.read_lineage(self.cfg, "G01", folder,
                                      allow_synthetic_test_only=True)[1],
                             line_rows)
            for label in ("hard_floor", "prompt_failures", "tokens",
                          "total_tokens", "peak_rss", "private_field"):
                with self.subTest(forgery=label):
                    bad = copy.deepcopy(manifest)
                    if label == "hard_floor":
                        bad["members"][0]["hard_floor_pass"] = False
                    elif label == "prompt_failures":
                        bad["members"][0]["hard_floor_failures"] = ["F01"]
                    elif label == "tokens":
                        bad["members"][0]["generated_tokens"] += 1
                    elif label == "total_tokens":
                        bad["total_generated_tokens"] += 1
                    elif label == "peak_rss":
                        bad["peak_rss_kib"] = 0
                    else:
                        bad["private_member_mapping"] = {"R001": "V-G01-A"}
                    assemble.write_json(archive_manifest, bad)
                    with self.assertRaisesRegex(
                            screen.ScreenError, "lineage .* drift"):
                        assemble.read_lineage(self.cfg, "G01", folder,
                                      allow_synthetic_test_only=True)
            assemble.write_json(archive_manifest, manifest)
            assemble.read_lineage(self.cfg, "G01", folder,
                                      allow_synthetic_test_only=True)


    def test_refused_rerun_preserves_old_receipts_and_work_bytes(self):
        cases = ("receipt_only", "work_only", "complete")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                out = Path(td) / "lineage"
                out.mkdir()
                if case == "receipt_only":
                    (out / "interrupted.json").write_bytes(
                        b'{"old":"failure-original"}\n')
                elif case == "work_only":
                    work = out / "work" / "V-G01-A"
                    work.mkdir(parents=True)
                    (work / "adapter_model.safetensors").write_bytes(b"original")
                else:
                    for name in ("manifest.json", "results.jsonl",
                                 "results.partial.jsonl", "interrupted.json"):
                        (out / name).write_bytes(
                            ("ORIGINAL " + name).encode("utf-8"))
                before = {str(p.relative_to(out)): p.read_bytes()
                          for p in out.rglob("*") if p.is_file()}
                with mock.patch.object(sys, "argv", [
                    "live.py", "--run", "--lineage", "G01",
                    "--out", str(out)]), mock.patch.object(
                        live, "run_lineage",
                        side_effect=AssertionError("must not enter model path")
                    ) as model:
                    with self.assertRaisesRegex(
                            live.PreexistingLineageEvidence, "refuse"):
                        live.main()
                    model.assert_not_called()
                after = {str(p.relative_to(out)): p.read_bytes()
                         for p in out.rglob("*") if p.is_file()}
                self.assertEqual(before, after)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "first-failure"
            with mock.patch.object(sys, "argv", [
                "live.py", "--run", "--lineage", "G01",
                "--out", str(out)]), mock.patch.object(
                    live, "run_lineage", side_effect=RuntimeError("first-error")
                    ):
                with self.assertRaisesRegex(RuntimeError, "first-error"):
                    live.main()
            receipt_path = out / "interrupted.json"
            original = receipt_path.read_bytes()
            with mock.patch.object(sys, "argv", [
                "live.py", "--run", "--lineage", "G01",
                "--out", str(out)]), mock.patch.object(
                    live, "run_lineage",
                    side_effect=AssertionError("must not rerun")) as model:
                with self.assertRaises(live.PreexistingLineageEvidence):
                    live.main()
                model.assert_not_called()
            self.assertEqual(receipt_path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                live.write_interrupted_receipt(
                    "G01", out, RuntimeError("second-error"))
            self.assertEqual(receipt_path.read_bytes(), original)

    def test_atomic_lineage_claim_refuses_empty_and_interleaved_reruns(self):
        """One writer owns the output even when both preflights see it absent."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty = root / "empty-existing-output"
            empty.mkdir()
            with self.assertRaisesRegex(
                    live.PreexistingLineageEvidence, "pre-existing"):
                live.run_lineage("G01", empty, root / "downloads")
            self.assertEqual(list(empty.iterdir()), [])

            # Deterministic interleaving: main() preflight saw an absent
            # target, then another process created its directory/receipt just
            # before run_lineage() attempts the exclusive mkdir. No inference,
            # receipt overwrite, or partial append is permitted.
            raced = root / "interleaved"
            original = b'{"owner":"first-worker","error":"original"}\n'
            passes = []

            def simulate_competitor(path):
                passes.append(path)
                if len(passes) == 2:
                    path.mkdir(parents=True, exist_ok=False)
                    (path / "interrupted.json").write_bytes(original)

            with mock.patch.object(
                    live, "refuse_existing_lineage_evidence",
                    side_effect=simulate_competitor), mock.patch.object(
                    sys, "argv", [
                        "live.py", "--run", "--lineage", "G01",
                        "--out", str(raced)]):
                with self.assertRaisesRegex(
                        live.PreexistingLineageEvidence, "concurrent"):
                    live.main()
            self.assertEqual(len(passes), 2)
            self.assertEqual((raced / "interrupted.json").read_bytes(), original)
            self.assertEqual(
                [p.name for p in raced.iterdir()], ["interrupted.json"])

    def test_live_incremental_raw_and_failure_receipt_are_custodied(self):
        rows = self.synthetic_rows()[:2]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            partial = out / "results.partial.jsonl"
            for row in rows:
                live.append_completed_raw(partial, row)
            self.assertEqual(screen.read_results(partial), rows)
            receipt = live.write_interrupted_receipt(
                "G01", out, RuntimeError("synthetic failure"))
            self.assertEqual(receipt["completed_raw_results"], 2)
            self.assertEqual(receipt["partial_results_sha256"],
                             screen.sha256_file(partial))
            self.assertEqual(receipt["status"], "incomplete_artifact_only")
            self.assertFalse(receipt["durably_promoted"])
            self.assertFalse(receipt["population_expansion_admitted"])
            self.assertEqual(
                screen.read_json(out / "interrupted.json"), receipt)

    def test_interrupted_receipt_keeps_torn_or_corrupt_raw_bytes(self):
        row = self.synthetic_rows()[0]
        cases = (
            (b'{"torn":', "unterminated_raw_row"),
            (bytes([255, 10]), "unparseable_raw_row"),
            (b'null\n', "non_object_raw_row"),
        )
        for tail, expected_problem in cases:
            with self.subTest(problem=expected_problem):
                with tempfile.TemporaryDirectory() as td:
                    out = Path(td)
                    partial = out / "results.partial.jsonl"
                    live.append_completed_raw(partial, row)
                    with partial.open("ab") as fh:
                        fh.write(tail)
                    original = partial.read_bytes()
                    receipt = live.write_interrupted_receipt(
                        "G01", out, RuntimeError("synthetic interruption"))
                    self.assertEqual(receipt["completed_raw_results"], 1)
                    self.assertEqual(receipt["partial_journal_problem"],
                                     expected_problem)
                    self.assertEqual(receipt["partial_results_sha256"],
                                     screen.sha256_file(partial))
                    self.assertEqual(partial.read_bytes(), original)
                    self.assertEqual(screen.read_json(out / "interrupted.json"),
                                     receipt)

    def test_live_workflow_is_manual_confirmed_main_only_and_isolated(self):
        workflow = (
            screen.ROOT / ".github/workflows/detective-005a-live.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("\npush:", workflow)
        self.assertIn("RUN-144-NO-TRAINING", workflow)
        self.assertIn("refs/heads/main", workflow)
        self.assertIn("max-parallel: 6", workflow)
        self.assertIn("!out/${{ matrix.lineage }}/work/**", workflow)
        self.assertIn("pattern: detective005a-G*", workflow)
        self.assertIn("TARGET_BRANCH: evidence/detective-005a-${{ github.run_id }}", workflow)
        self.assertIn("git merge-base --is-ancestor", workflow)
        self.assertIn("Evidence branch already exists; refusing overwrite.", workflow)
        self.assertNotIn("TARGET_BRANCH: research/detective-005a-fun-screen-20261028", workflow)


if __name__ == "__main__":
    unittest.main()
