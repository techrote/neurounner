#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from corpus import (ACTIONS, SCENARIOS, SPLITS, episode, candidate_state,
                    build, audit, partition, exact_or_near_duplicate, world)
from ingest_w0 import ingest


class GameCorpusTests(unittest.TestCase):
    def test_all_scenarios_and_legal_actions(self):
        for scenario in SCENARIOS:
            w, ep, rows=episode(345019,scenario,'pipework',budget=40)
            self.assertTrue(rows)
            self.assertEqual(w['size'],96)
            self.assertEqual(rows[0]['observation']['heading'],0)
            self.assertTrue(all(len(r['observation']['local_5x5'])==5 for r in rows))
            self.assertTrue(all(r['target_action'] in r['observation']['legal'] for r in rows))
            self.assertTrue(all(r['receipt']['raw_legal'] for r in rows))
            self.assertTrue(all(r['receipt']['position_after'] == ep['trail'][i+1] for i,r in enumerate(rows)))
            prohibited={tuple(p) for p in w['protected']}
            obstacles={tuple(p) for p in w['obstacles']}
            self.assertFalse(prohibited.intersection(tuple(p) for p in ep['trail']))
            self.assertFalse(obstacles.intersection(tuple(p) for p in ep['trail']))

    def test_exact_determinism_and_matching_families(self):
        a=episode(87642,'motif','greebles',budget=58)
        b=episode(87642,'motif','greebles',budget=58)
        c=episode(87642,'motif','architecture',budget=58)
        self.assertEqual(a,b)
        self.assertEqual(a[0]['world_id'],c[0]['world_id'])
        self.assertEqual(a[0]['split'],c[0]['split'])
        self.assertNotEqual(a[1]['episode_id'],c[1]['episode_id'])

    def test_refused_paths(self):
        obs={(5,4),(6,5)}
        protected={(6,4)}
        moves,fresh,legal=candidate_state((5,5),0,{(5,5)},[],obs,protected,96)
        self.assertNotIn('STRAIGHT',legal)
        self.assertNotIn('LEFT',legal)
        self.assertIn('RIGHT',legal)
        self.assertNotIn('BRANCH',legal)
        self.assertNotIn('RETRACE',legal)
        self.assertIn('STOP',legal)

    def test_family_and_hard_negative_semantics(self):
        _,_,rows=episode(777,'deceptive','greebles',budget=40)
        for r in rows:
            self.assertEqual(r['target_origin'],'deterministic-synthetic-heuristic-not-native')
            for n in r['hard_negatives']:
                self.assertIn(n['action'],ACTIONS)
                if n['kind']=='host_illegal':
                    self.assertNotIn(n['action'],r['observation']['legal'])
                else:
                    self.assertIn(n['action'],r['observation']['legal'])
                    self.assertNotEqual(n['action'],r['target_action'])

    def test_cross_split_world_population_and_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m=build(root,worlds_per_scenario=10,budget=16,png_limit=6)
            self.assertEqual(m['counts']['worlds'],{'train':48,'dev':6,'heldout':6})
            self.assertEqual(m['counts']['episodes'],{'train':144,'dev':18,'heldout':18})
            self.assertEqual(audit(root)['audit'],'PASS')
            self.assertEqual(sum(s.endswith('.png') for s in m['files']),6)
            groups={s:{json.loads(line)['world_group_id'] for line in (root/f'{s}.worlds.jsonl').read_text().splitlines()} for s in SPLITS}
            self.assertFalse(groups['train']&groups['dev'])
            self.assertFalse(groups['train']&groups['heldout'])
            self.assertFalse(groups['dev']&groups['heldout'])
            manifest=(root/'manifest.json').read_bytes()
            again=Path(d)/'another';build(again,worlds_per_scenario=10,budget=16,png_limit=6)
            self.assertEqual(manifest,(again/'manifest.json').read_bytes())
            bad=root/'train.decisions.jsonl';bad.write_bytes(bad.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                audit(root)

    def test_explicitly_bounded(self):
        with self.assertRaises(ValueError):episode(1,'motif','greebles',budget=9999)
        with self.assertRaises(ValueError):world(1,'nosuch')
        with self.assertRaises(ValueError):build(Path('/nonexistent/irrelevant'),worlds_per_scenario=2)


if __name__=='__main__':unittest.main()
