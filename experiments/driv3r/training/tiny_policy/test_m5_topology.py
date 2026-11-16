"""M5 source-local regression: splits, STOP, host legality and audit failures."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .core import ACTIONS,DELTA,FAMILIES
from .m5_topology_train import (PROTOCOL,TRAIN_SEEDS,DEV_SEEDS,EVAL_SEEDS,M4_SEEDS,
                                 TURNS,maze_world,oracle,mixed_examples,geometry_sha,validated_m3_lineage)
from .audit_m5_topology import audit


class M5ContractTests(unittest.TestCase):
    def test_disjoint_seed_identities(self):
        parts=[set(TRAIN_SEEDS),set(DEV_SEEDS),set(EVAL_SEEDS),set(M4_SEEDS)]
        for i,a in enumerate(parts):
            for b in parts[i+1:]:self.assertFalse(a.intersection(b))

    def test_observation_and_family_independent_geometry(self):
        for seed in (TRAIN_SEEDS[0],TRAIN_SEEDS[1],DEV_SEEDS[0],EVAL_SEEDS[0]):
            worlds=[maze_world(seed,f) for f in FAMILIES]
            self.assertEqual(len({geometry_sha(w) for w in worlds}),1)
            for w in worlds:
                self.assertEqual(w.observe().shape,(162,))
                self.assertTrue(any(a!=4 for a in w.legal()))
                self.assertEqual(w.budget,96)

    def test_oracle_legal_across_dead_end_retrace(self):
        for seed in TRAIN_SEEDS[:4]:
            for family in FAMILIES:
                w=maze_world(seed,family)
                for _ in range(TURNS):
                    legal=w.legal()
                    choice=oracle(w)
                    self.assertIn(choice,legal)
                    receipt=w.apply(choice)
                    self.assertIn(receipt['action'],ACTIONS)
                    if w.stopped:break

    def test_protected_stop_is_real(self):
        w=maze_world(TRAIN_SEEDS[0],FAMILIES[0])
        for dx,dy in DELTA[:4]:
            w.protected[w.y+dy,w.x+dx]=True
        self.assertEqual(w.legal(),(4,))
        self.assertEqual(oracle(w),4)
        w.apply(4)
        self.assertTrue(w.stopped)
        self.assertFalse(w.legal())

    def test_synthetic_train_data_labels(self):
        data,meta=mixed_examples(TRAIN_SEEDS[:2],turns=12)
        xx,yy,mm,weights=data
        self.assertEqual(xx.shape[1],162)
        self.assertTrue(mm[np.arange(len(yy)),yy].all())
        self.assertEqual(meta['world_seeds'],list(TRAIN_SEEDS[:2]))
        self.assertIn('STOP',meta['actions'])
        repeat,meta2=mixed_examples(TRAIN_SEEDS[:2],turns=12)
        self.assertEqual(meta['identity_sha256'],meta2['identity_sha256'])
        self.assertTrue(np.array_equal(data[0],repeat[0]))

    def test_rebuilt_m3_needs_source_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/"checkpoints";ck.mkdir()
            with self.assertRaises((FileNotFoundError,ValueError)):
                validated_m3_lineage(ck,source_rebuilt=True)

    def test_forged_rebuilt_m3_lineage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ck=root/"checkpoints";ck.mkdir()
            for model in ("linear","mlp","conv"):(ck/(model+".pt")).write_bytes(b"fake checkpoint")
            (root/"report.json").write_text(json.dumps({"protocol":"tiny-policy-m3-curriculum-cpu-v1",
                "synthetic_only":True,"illegal_applied_actions":0,
                "models":{name:{"head_checkpoint_sha256":"not-the-model"} for name in ("linear","mlp","conv")}}))
            with self.assertRaisesRegex(ValueError,"disagrees"):
                validated_m3_lineage(ck,source_rebuilt=True)

    def test_fails_closed_on_missing_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError,ValueError)):
                audit(Path(tmp),Path(tmp))

    def test_fails_closed_on_forged_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'report.json').write_text(json.dumps({
                'protocol':'native-final-authorized','synthetic_only':False}))
            with self.assertRaisesRegex(ValueError,'non-synthetic'):
                audit(root,root)


if __name__=='__main__':
    unittest.main()
