"""Cheap offline guards: NO inference, native execution, paid calls or downloads."""
import unittest

from experiments.driv3r.backend import PINS
from experiments.driv3r.wormbench.reference import canonical, EXPECTED_DIGEST, SOURCE_SHA
from experiments.driv3r.wormbench.shadow import (
    MODELS, MAX_ROWS, parse_chat_choice, choose_chat_prompt, verify_request, legal_id_grammar,
)

class WormBenchTests(unittest.TestCase):
    def test_grammar_is_exact_and_cannot_expand_native_legal_candidates(self):
        self.assertEqual(legal_id_grammar(["MOVE_N","BACKTRACK","STOP"]),
                         'root ::= "MOVE_N" | "BACKTRACK" | "STOP"')
        for invalid in (["MOVE_N","MOVE_N"],["MOVE_N"],["MOVE_N","MOVE_S + 1"],
                        ["MOVE_N","MOVE_W|STOP"]):
            with self.assertRaises(ValueError):
                legal_id_grammar(invalid)

    def test_source_matched_smol_replacement_and_separate_sprocket_repair_arm(self):
        self.assertEqual(MODELS["smol135"]["repository"],
                         "QuantFactory/SmolLM2-135M-Instruct-GGUF")
        self.assertEqual(MODELS["smol135"]["filename"],
                         "SmolLM2-135M-Instruct.Q4_K_M.gguf")
        self.assertEqual(MODELS["smol135"]["kind"],"chat")
        self.assertEqual(MODELS["smol135-grammar"]["kind"],"chat_grammar")
        self.assertEqual(MODELS["sprocket500"]["kind"],"chat")
        self.assertEqual(MODELS["sprocket500-grammar"]["kind"],"chat_grammar")

    def test_q4_is_exact_optional_pin_and_never_default(self):
        self.assertEqual(PINS["model"]["quantization"],"Q8_0")
        self.assertTrue(PINS["optional_q4"]["filename"].endswith("-Q4_K_M.gguf"))
        self.assertEqual(PINS["optional_q4"]["sha256"],
            "adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b")
        self.assertNotEqual(PINS["model"]["sha256"],PINS["optional_q4"]["sha256"])
    def test_candidate_parse_accepts_only_exact_legal_ids(self):
        ids=["MOVE_N","MOVE_E","BACKTRACK","STOP"]
        for s in ids:
            self.assertEqual(parse_chat_choice(s,ids),s)
            self.assertEqual(parse_chat_choice('  "'+s+'" \n',ids),s)
        for invalid in ["MOVE_S","MOVE_N and MOVE_E","I pick MOVE_N",
                        '{"choice":"MOVE_N"}',"",'{"choice":"MOVE_S"}']:
            self.assertIsNone(parse_chat_choice(invalid,ids))
    def test_shadow_prompt_contains_only_recoverable_legal_options(self):
        row={"candidate_ids":["MOVE_N","STOP"],
             "questions":{"worm_move":{"criteria":{"MOVE_N":"go north","STOP":"finish"}}},
             "state":{"position":[4,5],"heading":"N","recent_positions":[[4,6]],
                      "mission":"draw a line","candidate_hints":{"MOVE_N":{"soft_scent_u16":5}},
                      "native_world_material_sha256":"f"*64}}
        prompt=choose_chat_prompt(row)
        self.assertIn("MOVE_N: go north",prompt)
        self.assertIn("STOP: finish",prompt)
        self.assertNotIn("f"*64,prompt)
        self.assertIn("soft_scent_u16",prompt)
    def test_owner_gate_rejects_model_and_scope_mutation(self):
        import hashlib
        rows=[{"mode":"free","turn":n} for n in range(MAX_ROWS)]
        digest=hashlib.sha256(canonical(rows)).hexdigest()
        request={"schema":"driv3r-slmworm-shadow/1",
                 "eligible_models":list(MODELS),
                 "allow_training":False,"allow_native":False,
                 "allow_paid_api":False,"max_shadow_states_per_model":MAX_ROWS,
                 "original_q8_artifact_digest":EXPECTED_DIGEST,
                 "owner_authorization":"Try d1 Q4 performance and bring in SLMworms"}
        doc={"digest":EXPECTED_DIGEST,"source_sha":SOURCE_SHA,"rows":rows}
        with self.assertRaises(ValueError):
            verify_request(request,"invalid","abc",doc)
        request["allow_native"]=True
        with self.assertRaises(ValueError):
            verify_request(request,"d1-q4","abc",doc)
        request["allow_native"]=False
        with self.assertRaises(ValueError):
            verify_request(request,"d1-q4","abc",doc) # synthetic rows lack frozen SHA
        self.assertNotEqual(digest,EXPECTED_DIGEST)

if __name__=="__main__":
    unittest.main()
