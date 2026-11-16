"""Fresh Allele independence and Garden counterfactual-compiler contracts."""
import unittest
from experiments.memplant.dreamgraft002.source import make_cases
from experiments.memplant.memorygarden.fresh_allele_source import cases as fresh_cases
from experiments.memplant.memorygarden.garden import World, Invention, evaluate
from experiments.memplant.memorygarden.garden_compiler import (
    parse_authenticated_laws, compile_design, constraints)


class FreshAlleleSources(unittest.TestCase):
    def test_no_original_worlds_or_templates_reused(self):
        new=fresh_cases()
        old=make_cases()
        self.assertEqual(len(new),32)
        self.assertEqual(len({x["family"] for x in new}),8)
        old_names={x["family"] for x in old}
        old_sources={x["source_sha256"] for x in old}
        self.assertTrue(all(x["family"] not in old_names and
                            x["source_sha256"] not in old_sources for x in new))
        self.assertEqual({x["answer_class"] for x in new},{0,1,2,3})
        for x in new:
            for word in ("hardens","softens","floats","sinks"):
                self.assertEqual(x["source"].count(word),1)
        idmap={x["case_id"]:x for x in new}
        for x in new:
            h=idmap[x["heat_twin"]]
            v=idmap[x["vibe_twin"]]
            self.assertEqual(h["family"],x["family"])
            self.assertNotEqual(h["heat"],x["heat"])
            self.assertEqual(h["vibration"],x["vibration"])
            self.assertNotEqual(v["vibration"],x["vibration"])
            self.assertEqual(v["heat"],x["heat"])


class MemoryCompilerTests(unittest.TestCase):
    def test_card_and_counterfactual_alter_all_required_decisions(self):
        raw=Invention("Glass Reef","rigid","ceiling","heat_then_shake","Model idea")
        w=World("w",True,True)
        compiled,trace=compile_design(raw,evidence=w.postcard(),kind="card")
        opposite,other=compile_design(raw,evidence=w.inverted().postcard(),kind="card")
        self.assertTrue(evaluate(w,compiled)["passed"])
        self.assertTrue(evaluate(w.inverted(),opposite)["passed"])
        self.assertFalse(evaluate(w,opposite)["passed"])
        self.assertEqual(set(k for k in ("frame","collector","sequence")
                             if getattr(compiled,k)!=getattr(opposite,k)),
                         {"frame","collector","sequence"})
        self.assertTrue(other["repair_applied"])
        self.assertEqual(other["compiler"],
                         "deterministic rule-to-mechanics, not LLM reasoning")

    def test_raw_no_memory_is_not_privileged_or_repaired(self):
        raw=Invention("Mystery","rigid","ceiling","heat_then_shake","Model guess")
        finished,trace=compile_design(raw)
        self.assertEqual(finished,raw)
        self.assertFalse(trace["memory_available"])
        self.assertFalse(trace["repair_applied"])

    def test_full_source_parser_uses_verified_law_not_hidden_labels(self):
        original=make_cases()
        for x in original:
            if x["family"] not in ("two-rule-26","two-rule-27"):
                continue
            w=parse_authenticated_laws(x["source"],kind="source",name=x["case_id"])
            self.assertEqual(w.heat,x["heat"])
            self.assertEqual(w.vibration,x["vibration"])
        with self.assertRaises(ValueError):
            parse_authenticated_laws("No verified rule here",kind="source",name="none")
        with self.assertRaises(ValueError):
            parse_authenticated_laws(
                "Recovered field observation: warming the alloy hardens it. "
                "Vibrating its dust makes it floats. SECRET LABEL",
                kind="card",name="bad")

if __name__=="__main__":
    unittest.main()
