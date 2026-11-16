"""M3 fail-closed source-contract, split, STOP and evidence tests."""
import copy
import random
import unittest
from .core import ACTIONS, FAMILIES
from .m3_curriculum import (BUDGET, CASE_NAMES, SPLITS, examples, geometry_hash,
                             oracle, replay, rollout, world)

class M3Tests(unittest.TestCase):
    def test_contract_and_disjoint_seed_groups(self):
        assert len(set(sum((list(v) for v in SPLITS.values()), [])))==sum(map(len,SPLITS.values()))
        self.assertEqual(ACTIONS,('UP','DOWN','LEFT','RIGHT','STOP'))
        self.assertEqual(BUDGET,96)
        self.assertEqual(len(CASE_NAMES),4)
        for seed in (7300,7301,7302,7303,9300,9301,9302,9303):
            hashes=[geometry_hash(world(seed,family)) for family in FAMILIES]
            self.assertEqual(len(set(hashes)),1)

    def test_real_protected_trap_has_only_stop(self):
        for seed in (7300,7301,7302,7303):
            w=world(seed,'Greebles',trapped=True)
            self.assertEqual(w.legal(),(4,))
            self.assertEqual(oracle(w,random.Random(13)),4)
            self.assertEqual(w.apply(4)['action'],'STOP')
            self.assertTrue(w.stopped)

    def test_oracle_legal_and_replay(self):
        for seed in (9300,9301,9302,9303):
            for fam in FAMILIES:
                w=world(seed,fam)
                e=rollout(w,oracle,seed,steps=32)
                self.assertEqual(e['unique'],int(replay(e).visited.sum()))
                self.assertEqual(e['budget'],96)
                self.assertTrue(all(x['action'] in x['legal'] for x in e['events']))

    def test_corrupt_event_and_provenance_rejected(self):
        w=world(9302,'Pipework')
        e=rollout(w,oracle,9302,steps=10)
        broken=copy.deepcopy(e);broken['events'][0]['action']='NOT_AN_ACTION'
        with self.assertRaises(ValueError):replay(broken)
        broken=copy.deepcopy(e);broken['geometry_sha256']='0'*64
        with self.assertRaises(ValueError):replay(broken)
        broken=copy.deepcopy(e);broken['events'][0]['feature_sha256']='0'*64
        with self.assertRaises(ValueError):replay(broken)
        broken=copy.deepcopy(e);broken['window']=129
        with self.assertRaises(ValueError):replay(broken)

    def test_examples_stop_labels_only_in_forced_stops(self):
        x,y,mask,weights,ids=examples((7300,),turns=8)
        self.assertEqual(x.shape[1],162)
        self.assertEqual(len(y),len(ids))
        self.assertEqual(sum(y==4),3)
        self.assertTrue(all(mask[k,label] for k,label in enumerate(y)))
        self.assertTrue(all(weights>0))
        self.assertTrue(all(identity[0]==7300 for identity in ids))

if __name__=='__main__': unittest.main()
