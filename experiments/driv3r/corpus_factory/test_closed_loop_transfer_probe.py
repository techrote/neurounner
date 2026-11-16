"""Focused source/host-safety regressions for cross-generator synthetic inference."""
import json
import os
from pathlib import Path
import unittest

from closed_loop_transfer_probe import (make_world,legal_state,observation,rollout,train_model,sha,SOURCE_SHA)

BASE=Path(os.environ.get('CORPUS_ARTIFACT_DIR','/mnt/data'))

class TransferProbeTest(unittest.TestCase):
    def test_seed_determinism_reachable(self):
        a,b=make_world(7),make_world(7)
        self.assertEqual(a,b)
        self.assertGreaterEqual(a['shortest_goal'],38)
        self.assertNotIn(a['goal'],a['protected'] | a['obstacles'])

    def test_immutable_protected_and_host_legal(self):
        w=make_world(13)
        r=rollout(w,'greebles','random',budget=128)
        self.assertEqual(r['illegal'],0)
        self.assertEqual(r['steps'],128)
        self.assertLessEqual(r['visited'],129)
        self.assertEqual(len(r['path']),129)
        self.assertTrue(all(p not in w['protected'] | w['obstacles'] for p in r['path']))

    def test_stop_never_illegal(self):
        w=make_world(3)
        moves,fresh,legal=legal_state(w,(7,48),0,{(7,48)},[])
        self.assertIn('STOP',legal)
        self.assertNotIn('RETRACE',legal)
        obs=observation(w,(7,48),0,{(7,48)},[],legal,0,128,'pipework')
        self.assertEqual(obs['local_5x5'][2][2],'@')
        self.assertEqual(obs['goal_delta'],[w['goal'][0]-7,w['goal'][1]-48])

    def test_bad_source_pin_refused(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p=Path(d)/'tampered.zip';p.write_bytes(b'not a source ZIP')
            with self.assertRaisesRegex(ValueError,'SHA pin'):
                train_model(p,'v1')

    def test_source_pins_present(self):
        for kind,p in [('v1',BASE/'cybersand_game_corpus_v1_train_only.zip'),
                       ('probe',BASE/'cybersand_corpus_family_contrast_probe_v1.zip')]:
            if not p.exists():
                self.assertEqual(len(SOURCE_SHA[kind]),64) # ZIP absent on source-only CI
            else:
                self.assertEqual(sha(p.read_bytes()),SOURCE_SHA[kind])

if __name__=='__main__':unittest.main()
