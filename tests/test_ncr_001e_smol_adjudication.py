"""SmolLM2 no-model archived-evidence negative refusal suite."""
import copy, json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from ncr_001e_smol_adjudication import OLD,LONG,PROTOCOL,REPORT,build,verify_old,canonical,check_archive

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(OLD/"receipt.json").read_bytes()
        cls.old=json.loads(cls.raw);cls.om=json.loads((OLD/"provenance.json").read_bytes())
        cls.prod=json.loads((LONG/"producer.json").read_bytes())
        cls.cons=json.loads((LONG/"consumer.json").read_bytes())
        cls.nm=json.loads((LONG/"provenance.json").read_bytes())
        cls.proto=json.loads(PROTOCOL.read_bytes())
    def build(self,proto=None,prod=None,cons=None):
        return build(self.proto if proto is None else proto,self.old,
            self.prod if prod is None else prod,self.cons if cons is None else cons,self.om,self.nm)
    def test_source_exact_machine_readback(self):
        report=check_archive()
        self.assertFalse(report["final_72_pass"])
        self.assertFalse(report["train_authorized"])
        self.assertEqual(REPORT.read_bytes(),canonical(report))
    def test_old_bytes_mutation_refusal(self):
        with self.assertRaisesRegex(ValueError,"old source"):
            verify_old(self.old,self.om,self.raw+b" ")
    def test_old_masked_batch_failure_refusal(self):
        altered=copy.deepcopy(self.old)
        altered["checks"]["zero_init_right_padding_batch2"]["max_abs"]=.3
        with self.assertRaisesRegex(ValueError,"padded batch"):
            verify_old(altered,self.om,self.raw)
    def test_old_cache_failure_refusal(self):
        altered=copy.deepcopy(self.old)
        altered["checks"]["nonzero_48_full_vs_incremental_KV"]["max_abs"]=.3
        with self.assertRaisesRegex(ValueError,"cache/causal"):
            verify_old(altered,self.om,self.raw)
    def test_old_variant_status_refusal(self):
        altered=copy.deepcopy(self.old);altered["status"]="PASS_FINAL_72"
        with self.assertRaisesRegex(ValueError,"variant"):
            verify_old(altered,self.om,self.raw)
    def test_training_flag_refusal(self):
        p=copy.deepcopy(self.proto);p["train_authorized"]=True
        with self.assertRaisesRegex(ValueError,"admission"):
            self.build(proto=p)
    def test_missing_learning_gate_refusal(self):
        p=copy.deepcopy(self.proto);p["conformance"]["required_checks"].remove("finite_state_and_loss")
        with self.assertRaisesRegex(ValueError,"check set"):
            self.build(proto=p)
    def test_fake_sprocket_variant_refusal(self):
        p=copy.deepcopy(self.prod);p["lane"]="sprocket_chat"
        with self.assertRaisesRegex(ValueError,"variant"):
            self.build(prod=p)
    def test_false_optimizer_step_refusal(self):
        p=copy.deepcopy(self.prod);p["optimizer_steps"]=1
        with self.assertRaisesRegex(ValueError,"training"):
            self.build(prod=p)
    def test_fake_cold_interpreter_refusal(self):
        c=copy.deepcopy(self.cons);c["process"]["pid"]=self.prod["process"]["pid"]
        with self.assertRaisesRegex(ValueError,"separate interpreter"):
            self.build(cons=c)

if __name__=="__main__":unittest.main()
