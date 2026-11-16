"""GAL-005 real host-contract and adversarial provenance tests."""
import json
from pathlib import Path
import tempfile
import unittest

from lab import FEATURES, World, HostRefusal, TinyPolicy, INK
from candidate_features import (CandidateWorld, SCHEMA, audit, AuditRefusal,
                                file_sha, paired, train_model, write_run)


class CandidateFeaturesTests(unittest.TestCase):
    def test_host_owned_menu_and_source_identity_never_change(self):
        samples=[CandidateWorld(240001,'pillars','sweep',v) for v in
                 ('legacy','candidate','candidate_no_open','candidate_no_ink')]
        offers=[w.offer() for w in samples]
        self.assertTrue(all(o[0]==offers[0][0] for o in offers))
        self.assertTrue(all(w.original==samples[0].original for w in samples))
        self.assertIn('STOP',offers[0][0])
        for w in samples:
            self.assertEqual(w.material,w.original)
            self.assertEqual(w.decisions,0)

    def test_candidate_features_differ_and_never_mutate_material(self):
        w=CandidateWorld(240000,'empty','sweep','candidate')
        w.own.update({(10,24),(10,25),(11,25)})
        before=(w.material.copy(),w.cursor,w.decisions,w.logical_moves,w.visits.copy())
        menu=w.legal();v={n:w.features(n,p) for n,p in menu.items()}
        self.assertGreater(len({round(x[FEATURES.index('near_ink')],5) for x in v.values()}),1)
        self.assertGreater(len({round(x[FEATURES.index('open')],5) for x in v.values()}),1)
        self.assertEqual(before,(w.material,w.cursor,w.decisions,w.logical_moves,w.visits))

    def test_endpoint_open_respects_protection_and_proposed_ink(self):
        w=CandidateWorld(240000,'empty','sweep','candidate')
        menu=w.legal();key='INK_E_1';end=menu[key][-1]
        opening=w.features(key,menu[key])[FEATURES.index('open')]
        adj=(end[0]+1,end[1]);w.protected.add(adj)
        w.material[w.idx(adj)]=5
        blocked=w.features(key,menu[key])[FEATURES.index('open')]
        self.assertEqual(round(opening-blocked,5),.25)
        with self.assertRaises(HostRefusal):w.apply(key,'forged')

    def test_legacy_near_and_open_constant_per_offer(self):
        w=CandidateWorld(240000,'empty','sweep','legacy');menu=w.legal()
        for k in ('near_ink','open'):
            self.assertEqual(len({w.features(n,p)[FEATURES.index(k)] for n,p in menu.items()}),1)

    def test_stop_and_wrong_world_still_refused(self):
        w=CandidateWorld(240000,'empty','sweep','candidate')
        menu,token=w.offer()
        with self.assertRaises(HostRefusal):w.apply('INK_E_999',token)
        with self.assertRaises(HostRefusal):w.apply('INK_E_1','fabricated')
        w.apply('STOP',token)
        with self.assertRaises(HostRefusal):w.apply('INK_E_1',token)

    def test_training_updates_weights(self):
        p,n=train_model('sweep','candidate',3)
        self.assertGreater(n,0)
        self.assertTrue(any(abs(v)>1e-9 for v in p.weights))
        with self.assertRaises(ValueError):train_model('sweep','candidate',161)

    def test_paired_custody_refusal(self):
        x={'seed':240000,'family':'empty','initial_material_sha256':'aaa',
           'first_32':None,'trace_sha256':'foo'}
        with self.assertRaises(ValueError):paired([x],[{**x,'initial_material_sha256':'bbb'}])

    def test_complete_train_replay_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);report=write_run(root,train_worlds=2,heldout_worlds=5,gallery_worlds=1)
            self.assertEqual(report['schema'],SCHEMA)
            audited=audit(root)
            self.assertEqual(audited['episodes_replayed'],40)
            self.assertEqual(len(list((root/'gallery').glob('*.png'))),8)
            traces=root/'traces.json';old=traces.read_text();traces.write_text(old+' ')
            with self.assertRaises(AuditRefusal):audit(root)
            traces.write_text(old)
            self.assertEqual(audit(root)['episodes_replayed'],40)
            cp=root/'checkpoints'/'sweep_candidate.json'
            cp.write_text(cp.read_text().replace('"weights"','"WElGHTS"'))
            with self.assertRaises(AuditRefusal):audit(root)

    def test_budget_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):write_run(temp,train_worlds=0)
            with self.assertRaises(ValueError):write_run(temp,heldout_worlds=81)
            with self.assertRaises(ValueError):write_run(temp,gallery_worlds=6)

if __name__=='__main__':unittest.main()
