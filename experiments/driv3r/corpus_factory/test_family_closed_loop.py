#!/usr/bin/env python3
"""Executable external synthetic corpus checks, including poisoned replay."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from family_closed_loop import (ACTIONS, FAMILIES, SCENARIOS, OBS_KEYS, audit,
                                canonical, digest, export_train_only, image_bytes,
                                make_world, produce, rollout)


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_closed_loop_seed_repeatability_and_host_legality(self):
        first=rollout(12816,'protected','greebles',budget=128)
        second=rollout(12816,'protected','greebles',budget=128)
        self.assertEqual(first,second)
        world,ep,rows=first
        self.assertEqual(len(rows),128)
        self.assertTrue(ep['terminated_by_stop'])
        self.assertEqual(rows[-1]['target_action'],'STOP')
        self.assertEqual(rows[-1]['receipt']['stop_reason'],'budget_bound')
        self.assertEqual(image_bytes(world,ep),image_bytes(second[0],second[1]))
        protected={tuple(p) for p in world['protected']}
        obstacles={tuple(p) for p in world['obstacles']}
        last=tuple(world['start'])
        for row in rows:
            self.assertEqual(tuple(row['receipt']['position_before']),last)
            self.assertIn(row['target_action'],row['observation']['legal'])
            self.assertTrue(row['receipt']['raw_legal'])
            last=tuple(row['receipt']['position_after'])
            self.assertNotIn(last,protected)
            self.assertNotIn(last,obstacles)
        self.assertEqual(last,tuple(ep['trail'][-1]))
        self.assertGreater(ep['unique_trail_cells'],1)

    def test_matched_randomness_and_label_only_ablation(self):
        seed=432199
        full=[rollout(seed,'motif',f,128)[1] for f in FAMILIES]
        self.assertGreater(len({digest(sorted({tuple(p) for p in e['trail']})) for e in full}),1)
        # Every family label uses the same stochastic stream and same fixed
        # pipework steering in the ablation. Merely naming the family does NOT
        # generate independent procedural randomness or divergent drawings.
        ablated=[rollout(seed,'motif',f,128,steering_family='pipework')[1] for f in FAMILIES]
        self.assertEqual(len({digest(e['trail']) for e in ablated}),1)

    def test_split_replay_and_train_only_messages(self):
        folder=self.root/'data'
        summary=produce(folder,worlds_per_scenario=10,budget=64,image_worlds=0)
        self.assertEqual(summary['counts']['world_groups'],60)
        self.assertEqual(summary['counts']['split_groups'],{'train':48,'dev':6,'heldout':6})
        self.assertEqual(summary['counts']['label_only_ablation_footprints'],{1:60})
        self.assertEqual(audit(folder)['audit'],'PASS')
        export=export_train_only(folder,self.root/'export',per_episode=8)
        self.assertGreater(export['rows'],50)
        self.assertEqual(export['split'],'train-only')
        self.assertEqual(export['model_updates'],0)
        messages=(self.root/'export'/'train.messages.jsonl').read_text().splitlines()
        provenance=(self.root/'export'/'train.provenance.jsonl').read_text().splitlines()
        self.assertEqual(len(messages),len(provenance))
        for row,p in zip(messages,provenance):
            msg=json.loads(row)['messages'];obs=json.loads(msg[1]['content'])
            self.assertEqual(set(obs),OBS_KEYS)
            self.assertIn(msg[2]['content'],obs['legal'])
            self.assertEqual(json.loads(p)['split'],'train')
            for key in ('receipt','hard_negatives','target_action','candidate_scores','world_id'):
                self.assertNotIn(key,obs)
        self.assertTrue((folder/'heldout.evaluation-eligible.jsonl').exists())
        self.assertFalse((self.root/'export'/'heldout.messages.jsonl').exists())
        with self.assertRaises(FileExistsError):
            export_train_only(folder,self.root/'export')

    def test_forged_rehashed_receipt_refused_by_source_replay(self):
        folder=self.root/'data'
        produce(folder,worlds_per_scenario=10,budget=32,image_worlds=0)
        eligible_path=folder/'train.eligible.jsonl'
        eligible_lines=eligible_path.read_text().splitlines()
        source=json.loads(eligible_lines[0]);record_id=source['record_id']
        paths=[folder/'train.receipts.jsonl',eligible_path]
        manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text())
        for path in paths:
            lines=path.read_text().splitlines()
            matched=False
            for i,line in enumerate(lines):
                row=json.loads(line)
                if row['record_id']==record_id:
                    row['receipt']['position_after']=[0,0]
                    lines[i]=canonical(row);matched=True;break
            self.assertTrue(matched)
            payload=('\n'.join(lines)+'\n').encode()
            path.write_bytes(payload)
            manifest['files'][path.name]['sha256']=hashlib.sha256(payload).hexdigest()
            manifest['files'][path.name]['bytes']=len(payload)
        manifest_path.write_text(canonical(manifest)+'\n')
        with self.assertRaisesRegex(ValueError,'replay mismatch'):
            audit(folder,replay=True)

    def test_forged_rehashed_train_label_rejected_from_eligible_copy(self):
        folder=self.root/'data'
        produce(folder,worlds_per_scenario=10,budget=32,image_worlds=0)
        path=folder/'train.eligible.jsonl'
        lines=path.read_text().splitlines()
        modified=json.loads(lines[0])
        alternatives=[a for a in modified['observation']['legal'] if a!=modified['target_action']]
        self.assertTrue(alternatives)
        modified['target_action']=alternatives[0]
        lines[0]=canonical(modified)
        payload=('\n'.join(lines)+'\n').encode()
        path.write_bytes(payload)
        manifest_path=folder/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['files'][path.name]['sha256']=hashlib.sha256(payload).hexdigest()
        manifest['files'][path.name]['bytes']=len(payload)
        manifest_path.write_text(canonical(manifest)+'\n')
        with self.assertRaisesRegex(ValueError,'eligible row not present identically'):
            audit(folder,replay=True)

    def test_bounds_and_rejected_bad_labels(self):
        with self.assertRaises(ValueError):
            rollout(1,'motif','unknown',128)
        with self.assertRaises(ValueError):
            rollout(1,'unknown','greebles',128)
        with self.assertRaises(ValueError):
            rollout(1,'motif','pipework',8)
        with self.assertRaises(ValueError):
            produce(self.root/'bad',worlds_per_scenario=9)


if __name__=='__main__':
    unittest.main()
