#!/usr/bin/env python3
"""Fail-closed model-first-pass sensitivity and source-label tests."""
import copy
import unittest
from stimulus_challenge_w7 import variants, score, sha, SCHEMA

OBS = {"family":"greebles","heading":0,"local_5x5":[".....","..#..","..@..",".....","....."],
       "legal":["STRAIGHT","LEFT","RIGHT","RETRACE","STOP"],
       "last_actions":["STRAIGHT"],"tick":12}

class StimulusTests(unittest.TestCase):
    def rows(self):
        return variants(copy.deepcopy(OBS), "seed3101:greebles")

    def answers(self, rows, code="0"):
        return [{"id":r["id"], "prompt_sha256":r["prompt_sha256"],"raw_text":code,
                 "token_ids":[901,7], "eos_terminated":True} for r in rows]

    def test_explicit_nonhost_counterfactuals(self):
        rows=self.rows()
        self.assertEqual([r["kind"] for r in rows],
                         ["observed","family_only","history_only","legal_subset"])
        self.assertEqual(rows[0]["observation"],OBS)
        self.assertEqual(rows[-1]["observation"]["legal"],["LEFT","RIGHT","RETRACE","STOP"])
        self.assertEqual(rows[-1]["removed_action"],"STRAIGHT")
        self.assertEqual(len({r["prompt_sha256"] for r in rows}),4)
        self.assertEqual(rows[1]["observation_origin"],"synthetic-prompt-edit-not-W0-executed")

    def test_constant_digit_has_zero_sensitivity_and_detected_legal_failure(self):
        rows=self.rows()
        s=score(rows,self.answers(rows))
        self.assertEqual(s["paired_stimuli"],3)
        self.assertEqual(s["by_perturbation"]["legal_subset"]["removed_base_action_cases"],1)
        self.assertEqual(s["by_perturbation"]["legal_subset"]["removed_base_action_cases_newly_legal"],0)
        self.assertEqual(s["by_perturbation"]["family_only"]["raw_action_changes"],0)
        self.assertEqual(s["per_kind"]["legal_subset"]["raw_legal"],0)

    def test_changed_model_action_not_equated_with_art_quality(self):
        rows=self.rows()
        first=self.answers(rows)
        first[-1]["raw_text"]="1"
        s=score(rows,first)
        self.assertEqual(s["by_perturbation"]["legal_subset"]["raw_action_changes"],1)
        self.assertEqual(s["by_perturbation"]["legal_subset"]["removed_base_action_cases_newly_legal"],1)
        self.assertIn("not automatically better art",s["warning"])

    def test_first_pass_custody_fails_closed(self):
        rows=self.rows()
        original=self.answers(rows)
        for alteration in ("duplicate","missing","eos_lie","prompt_switch","stray_id","noninteger"):
            with self.subTest(alteration=alteration):
                a=copy.deepcopy(original)
                if alteration=="duplicate":a.append(copy.deepcopy(a[-1]))
                if alteration=="missing":a.pop()
                if alteration=="eos_lie":a[0]["eos_terminated"]=False
                if alteration=="prompt_switch":a[0]["prompt_sha256"]="0"*64
                if alteration=="stray_id":a[0]["id"]="alien"
                if alteration=="noninteger":a[0]["token_ids"]=[True,7]
                with self.assertRaises(ValueError):score(rows,a)

    def test_model_refusal_preserved_without_repair(self):
        rows=self.rows()
        a=self.answers(rows,code="RIGHT")
        s=score(rows,a)
        self.assertEqual(s["per_kind"]["observed"]["raw_valid_eos"],0)
        self.assertEqual(s["per_kind"]["observed"]["raw_legal"],0)

if __name__=="__main__":unittest.main()
